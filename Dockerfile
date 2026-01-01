FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
# Line 6-10: Change aria2c to aria2
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make ffmpeg-free \
  cmake ninja-build aria2 tar xz vim nano dialog \
  libdrm-devel zlib-devel openssl-devel pgrep \
  numactl-devel gperftools-libs \
  && dnf clean all && rm -rf /var/cache/dnf/*

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx110X
RUN set -euo pipefail; \
  BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
  PREFIX="therock-dist-linux-${GFX}-all-${ROCM_MAJOR_VER}"; \
  KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
  | tr '<' '\n' \
  | grep -o "therock-dist-linux-${GFX}-all-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
  | sort -V | tail -n1)"; \
  if [ -z "$KEY" ]; then \
    echo "ERROR: No matching tarball found for prefix: ${PREFIX}"; \
    exit 1; \
  fi; \
  echo "Downloading Latest Tarball: ${KEY}"; \
  aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz; \
  mkdir -p /opt/rocm; \
  tar xzf therock.tar.gz -C /opt/rocm --strip-components=1; \
  rm therock.tar.gz

# 3. Configure Global ROCm Environment
# We add LD_PRELOAD for tcmalloc here to fix the shutdown crash
RUN export ROCM_PATH=/opt/rocm && \
  BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  printf '%s\n' \
  "export ROCM_PATH=/opt/rocm" \
  "export HIP_PLATFORM=amd" \
  "export HIP_PATH=/opt/rocm" \
  "export HIP_CLANG_PATH=/opt/rocm/llvm/bin" \
  "export HIP_DEVICE_LIB_PATH=$BITCODE_PATH" \
  "export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:\$PATH" \
  "export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:\$LD_LIBRARY_PATH" \
  "export ROCBLAS_USE_HIPBLASLT=1" \
  "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" \
  "export VLLM_TARGET_DEVICE=rocm" \
  "export HIP_FORCE_DEV_KERNARG=1" \
  "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1" \
  "export LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4" \
  > /etc/profile.d/rocm-sdk.sh && \
  chmod 0644 /etc/profile.d/rocm-sdk.sh

# 4. Python Venv Setup
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 5. Install PyTorch (TheRock Nightly)
RUN python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx110X-all/ \
  --pre torch torchaudio torchvision

WORKDIR /opt

# Flash-Attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
  cd flash-attention &&\
  git checkout main_perf &&\
  python setup.py install && \
  cd /opt && rm -rf /opt/flash-attention

# 6. Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm

# --- PATCHING ---
RUN echo "import sys, re" > patch_strix.py && \
  echo "from pathlib import Path" >> patch_strix.py && \
  # Patch 1: __init__.py
  echo "p = Path('vllm/platforms/__init__.py')" >> patch_strix.py && \
  echo "txt = p.read_text()" >> patch_strix.py && \
  echo "txt = txt.replace('import amdsmi', '# import amdsmi')" >> patch_strix.py && \
  echo "txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)" >> patch_strix.py && \
  echo "txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)" >> patch_strix.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_init()', 'pass')" >> patch_strix.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')" >> patch_strix.py && \
  echo "p.write_text(txt)" >> patch_strix.py && \
  # Patch 2: rocm.py
  echo "p = Path('vllm/platforms/rocm.py')" >> patch_strix.py && \
  echo "txt = p.read_text()" >> patch_strix.py && \
  echo "header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules[\"amdsmi\"] = MagicMock()\n'" >> patch_strix.py && \
  echo "txt = header + txt" >> patch_strix.py && \
  echo "txt = re.sub(r'device_type = .*', 'device_type = \"rocm\"', txt)" >> patch_strix.py && \  
  echo "txt = re.sub(r'device_name = .*', 'device_name = \"gfx1100\"', txt)" >> patch_strix.py && \ 
  echo "txt += '\n    def get_device_name(self, device_id: int = 0) -> str:\n        return \"AMD-gfx1100\"\n'" >> patch_strix.py && \
  echo "p.write_text(txt)" >> patch_strix.py && \
  echo "print('Successfully patched vLLM for Strix Halo')" >> patch_strix.py && \
  python patch_strix.py && \
  sed -i 's/gfx1200;gfx1201/gfx1100/' CMakeLists.txt

# Patch hipify.py to use hipify_torch (torch.utils.hipify doesn't exist in nightlies)
RUN python3 << 'EOPATCH'
from pathlib import Path

path = 'cmake/hipify.py'
with open(path) as f:
    content = f.read()

# Replace the import to handle both torch.utils.hipify and hipify_torch
content = content.replace(
    'from torch.utils.hipify.hipify_python import hipify',
    '''try:
    from torch.utils.hipify.hipify_python import hipify
except ModuleNotFoundError:
    from hipify_torch import hipify'''
)

with open(path, 'w') as f:
    f.write(content)

print("Patched cmake/hipify.py to use hipify_torch")
EOPATCH

# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11

# Install hipify_torch (required for vLLM ROCm build)
RUN python -m pip install git+https://github.com/ROCm/hipify_torch.git

# Mock offload-arch in PyTorch's bundled ROCm SDK to prevent build errors
RUN PYTORCH_ROCM_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")/_rocm_sdk_core && \
  if [ -d "$PYTORCH_ROCM_PATH" ]; then \
    find "$PYTORCH_ROCM_PATH" -name "offload-arch" -exec sh -c 'mv "$1" "$1.bak" 2>/dev/null || true' _ {} \; && \
    find "$PYTORCH_ROCM_PATH" -type d -name "bin" -exec sh -c 'printf "#!/bin/bash\necho gfx1100\nexit 0\n" > "$1/offload-arch" && chmod +x "$1/offload-arch"' _ {} \; ; \
  fi

ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV HIP_ARCHITECTURES="gfx1100"          
ENV AMDGPU_TARGETS="gfx1100"              
ENV MAX_JOBS="4"
ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV ROCM_TARGET_LST="gfx1100"
ENV GPU_ARCHS="gfx1100"

# --- CRITICAL FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

# Create a mock offload-arch script to avoid stderr being captured as compiler flags
# This replaces the real offload-arch which fails without a GPU present
RUN if [ -f /opt/rocm/llvm/bin/offload-arch ]; then mv /opt/rocm/llvm/bin/offload-arch /opt/rocm/llvm/bin/offload-arch.bak; fi && \
  printf '#!/bin/bash\necho "gfx1100" 2>/dev/null\nexit 0\n' > /opt/rocm/llvm/bin/offload-arch && \
  chmod +x /opt/rocm/llvm/bin/offload-arch

# Also create mock for rocm_agent_enumerator if it exists
RUN if [ -f /opt/rocm/bin/rocm_agent_enumerator ]; then \
  mv /opt/rocm/bin/rocm_agent_enumerator /opt/rocm/bin/rocm_agent_enumerator.bak && \
  printf '#!/bin/bash\necho "gfx1100" 2>/dev/null\nexit 0\n' > /opt/rocm/bin/rocm_agent_enumerator && \
  chmod +x /opt/rocm/bin/rocm_agent_enumerator; \
  fi

# Add PyTorch to CMAKE_PREFIX_PATH so CMake can find TorchConfig.cmake
RUN sed -i 's/execute_process(/execute_process(ERROR_QUIET /g' CMakeLists.txt || true

# Or more targeted:
RUN python3 << 'EOPATCH'
import re

path = 'CMakeLists.txt'
with open(path) as f:
    content = f.read()

# Add ERROR_QUIET to execute_process calls that might call offload-arch
content = re.sub(
    r'execute_process\s*\(\s*COMMAND\s+(.*?)OUTPUT_VARIABLE',
    r'execute_process(COMMAND \1ERROR_QUIET OUTPUT_VARIABLE',
    content
)

with open(path, 'w') as f:
    f.write(content)
    
print("Patched CMakeLists.txt")
EOPATCH

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  python3 -c "import torch, os, sys; sys.stdout.write(os.path.dirname(torch.__file__))" > /tmp/torch_path.txt 2>&1 && \
  export TORCH_INSTALL=$(cat /tmp/torch_path.txt | tail -1) && \
  export CMAKE_PREFIX_PATH="$TORCH_INSTALL/share/cmake/Torch;$TORCH_INSTALL" && \
  export CMAKE_HIP_ARCHITECTURES="gfx1100" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1100 -DHIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$TORCH_INSTALL/share/cmake/Torch" && \
  export VERBOSE=1 && \
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist . 2>&1 | tee /tmp/vllm_build.log || (tail -200 /tmp/vllm_build.log && exit 1) && \
  python -m pip install /tmp/dist/*.whl

# --- bitsandbytes (ROCm) ---
WORKDIR /opt
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git
WORKDIR /opt/bitsandbytes

# Explicitly set HIP_PLATFORM (Docker ENV, not /etc/profile)
ENV HIP_PLATFORM="amd"
ENV CMAKE_PREFIX_PATH="/opt/rocm"

# Force CMake to use the System ROCm Compiler (/opt/rocm/llvm/bin/clang++)
RUN cmake -S . \
  -DGPU_TARGETS="gfx1100" \
  -DBNB_ROCM_ARCH="gfx1100" \
  -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  && \
  make -j$(nproc) && \
  python -m pip install --no-cache-dir . --no-build-isolation --no-deps

# 8. Final Cleanup & Runtime
WORKDIR /opt
RUN chmod -R a+rwX /opt && \
  find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_vllm.py /usr/local/bin/start-vllm
COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py
RUN chmod 0644 /etc/profile.d/*.sh && chmod +x /usr/local/bin/start-vllm && chmod 0644 /opt/max_context_results.json
RUN chmod 0644 /etc/profile.d/*.sh
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

CMD ["/bin/bash"]
