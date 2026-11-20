#!/usr/bin/env bash
# CUDA compile check of the NEMO / linear-algebra Kokkos sources with clang (no GPU, no nvcc needed).
#
# Full device compilation (clang -x cuda -> PTX -> ptxas cubin) of every SU2 translation unit that
# contains Kokkos device code, against a Kokkos 5.2.2 CUDA build made with the same clang. This
# catches host-only calls from KOKKOS_FUNCTION / KOKKOS_LAMBDA, std:: containers in device code,
# virtual calls, missing device math, etc. It does NOT link or run anything.
#
# Assumptions / paths (override via env):
#   CLANG       clang++ with CUDA support (default /usr/bin/clang++-18; officially supports CUDA <= 12.3,
#               newer headers work with -Wno-unknown-cuda-version)
#   NV          pip-installed CUDA 12 wheels: nvidia-cuda-runtime-cu12, nvidia-cuda-cccl-cu12,
#               nvidia-cuda-nvcc-cu12 (ptxas + libdevice)
#   CUDA_FAKE   assembled toolkit dir (built here from $NV via symlinks, plus an empty curand stub
#               header, a concatenating fake 'fatbinary' and an empty libcuda.so stub)
#   KK_BUILD    Kokkos CUDA build dir (configured + kokkoscore built here if missing, ~10 min on 2 cores)
#   REF_BUILD   SU2 build dir whose compile_commands.json gives include paths / defines
#               (default build-kokkos-strict, a Serial Kokkos build)
#   ARCHS       GPU archs to check (default "sm_70 sm_90")
#   PTXAS_V=1   pass -v to ptxas (registers / stack / spills per kernel) -> $OUT/<arch>/<tu>.log
# Output: $OUT/<arch>/<tu>.log, summary on stdout. Exit 0 iff all TUs compile for all archs.
set -u
SU2=${SU2:-$(cd "$(dirname "$0")/../.." && pwd)}
CLANG=${CLANG:-/usr/bin/clang++-18}
NV=${NV:-/usr/local/lib/python3.13/site-packages/nvidia}
CUDA_FAKE=${CUDA_FAKE:-/tmp/cuda-fake}
KK_BUILD=${KK_BUILD:-/tmp/kokkos-cuda-b2}
REF_BUILD=${REF_BUILD:-$SU2/build-kokkos-strict}
ARCHS=${ARCHS:-"sm_70 sm_90"}
OUT=${OUT:-/tmp/cudacheck}
PTXAS_V=${PTXAS_V:-1}
# Second pass ("constexpr"): nvcc without --expt-relaxed-constexpr (neither Kokkos nor the dispatcher pass it)
# rejects calls to constexpr __host__ functions (std::min/max/abs, std::numeric_limits, ...) from device code, which
# clang accepts by default. Re-run the device-side Sema with that relaxation disabled and keep only errors located
# in SU2 sources (Kokkos' own std::move/std::forward uses are accepted by nvcc and are filtered out).
CONSTEXPR_PASS=${CONSTEXPR_PASS:-1}
NVCC_LIKE="-Xclang -fno-cuda-host-device-constexpr -D__CLANG_CUDA_TEXTURE_INTRINSICS_H__ -ferror-limit=0"
TUS=${TUS:-"SU2_CFD/src/fluid/kokkos/NEMOGasKokkos.cpp SU2_CFD/src/solvers/kokkos/CNEMOKokkosData.cpp
SU2_CFD/src/solvers/kokkos/CNEMOKokkosStages.cpp SU2_CFD/src/solvers/kokkos/CNEMOKokkosUpwind.cpp
SU2_CFD/src/solvers/kokkos/CNEMOKokkosResidual.cpp SU2_CFD/src/solvers/kokkos/CNEMOKokkosViscous.cpp
Common/src/linear_algebra/CSysMatrixKokkos.cpp Common/src/linear_algebra/CSysVectorKokkos.cpp
Common/src/linear_algebra/CSysVector.cpp Common/src/linear_algebra/CSysMatrix.cpp
Common/src/linear_algebra/CSysSolve.cpp SU2_CFD/src/solvers/CNEMOEulerSolver.cpp
SU2_CFD/src/solvers/CNEMONSSolver.cpp UnitTests/SU2_CFD/fluid/NEMOGasKokkos_validation.cpp"}

# ---- 1. fake CUDA toolkit ----
if [ ! -x "$CUDA_FAKE/bin/fatbinary" ]; then
  mkdir -p "$CUDA_FAKE"/{include,lib64/stubs,bin,nvvm}
  for d in "$NV"/cuda_runtime/include "$NV"/cuda_cccl/include "$NV"/cuda_nvcc/include; do
    for f in "$d"/*; do ln -sf "$f" "$CUDA_FAKE/include/"; done; done
  ln -sfn "$NV/cuda_nvcc/nvvm/libdevice" "$CUDA_FAKE/nvvm/libdevice"
  ln -sf "$NV/cuda_nvcc/bin/ptxas" "$CUDA_FAKE/bin/ptxas"
  for f in "$NV"/cuda_runtime/lib/*; do ln -sf "$f" "$CUDA_FAKE/lib64/"; done
  ln -sf libcudart.so.12 "$CUDA_FAKE/lib64/libcudart.so"
  echo '/* stub: curand not installed */' > "$CUDA_FAKE/include/curand_mtgp32_kernel.h"
  echo '{"cuda":{"name":"CUDA SDK","version":"12.9.1"}}' > "$CUDA_FAKE/version.json"
  echo 'void __fake_cuda(void){}' > "$CUDA_FAKE/fc.c"
  gcc -shared -fPIC -o "$CUDA_FAKE/lib64/stubs/libcuda.so" "$CUDA_FAKE/fc.c"
  cat > "$CUDA_FAKE/bin/fatbinary" <<'EOF'
#!/bin/bash
# fake fatbinary (compile-only): concatenate the cubin/ptx images into the --create file
out=""; ins=(); prev=""
for a in "$@"; do
  [ "$prev" = "--create" ] && out="$a"
  case "$a" in --create=*) out="${a#--create=}";; --image=*|--image3=*) ins+=("${a##*file=}");; esac
  prev="$a"
done
cat "${ins[@]}" > "$out"
EOF
  chmod +x "$CUDA_FAKE/bin/fatbinary"
fi

# ---- 2. Kokkos CUDA build with clang (arch only affects KokkosCore_config.h / libkokkoscore) ----
if [ ! -f "$KK_BUILD/KokkosCore_config.h" ]; then
  mkdir -p "$KK_BUILD" && (cd "$KK_BUILD" && cmake -G Ninja "$SU2/subprojects/Kokkos" \
    -DCMAKE_CXX_COMPILER="$CLANG" -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_VOLTA70=ON \
    -DKokkos_CUDA_DIR="$CUDA_FAKE" -DCUDAToolkit_ROOT="$CUDA_FAKE" \
    -DCMAKE_CXX_FLAGS=-Wno-unknown-cuda-version && ninja -j2 kokkoscore kokkoscontainers) || exit 2
fi

# ---- 3. compile every TU for every arch ----
st=0
for arch in $ARCHS; do
  mkdir -p "$OUT/$arch"
  for tu in $TUS; do
    name=$(basename "$tu" .cpp)
    args=$(python3 -I - "$REF_BUILD/compile_commands.json" "$tu" <<'EOF'
import json, shlex, sys
cc = json.load(open(sys.argv[1]))
e = next(e for e in cc if e['file'].endswith('/' + sys.argv[2]))
a = shlex.split(e['command'])[1:]
keep, skip = [], 0
for x in a:
    if skip: skip -= 1; continue
    if x in ('-MD',): continue
    if x in ('-MQ', '-MF', '-o'): skip = 1; continue
    if x in ('-c', '-march=native', '-std=c++17', '-fdiagnostics-color=always') or x.endswith('.cpp'): continue
    if '__CMake_build' in x: continue          # Serial Kokkos config headers -> replaced by CUDA ones
    keep.append(x)
print(' '.join(shlex.quote(x) for x in keep))
EOF
)
    pv=""; [ "$PTXAS_V" = 1 ] && pv="-Xcuda-ptxas -v"
    cmd="cd $REF_BUILD && $CLANG -x cuda --cuda-path=$CUDA_FAKE --cuda-gpu-arch=$arch -Wno-unknown-cuda-version -Wno-unknown-warning-option \
-I$KK_BUILD/core/src -I$KK_BUILD -I$KK_BUILD/containers/src -DKOKKOS_DEPENDENCE $args -std=c++20 $pv \
-c $SU2/$tu -o $OUT/$arch/$name.o"
    echo "$cmd" > "$OUT/$arch/$name.cmd"
    if (eval "$cmd") > "$OUT/$arch/$name.log" 2>&1; then r=OK
    elif grep -q 'Segmentation fault' "$OUT/$arch/$name.log" && grep -q 'x86_64-pc-linux-gnu -target-sdk' "$OUT/$arch/$name.log"; then
      # clang 18 x86 *host* backend crash (seen on CNEMOEulerSolver::BC_Sym_Plane, any -O level). Under nvcc the
      # host side is g++, so check device codegen (full, to cubin) + host-side CUDA semantics (-fsyntax-only).
      mv "$OUT/$arch/$name.log" "$OUT/$arch/$name.hostcrash.log"
      dcmd="${cmd/ -c / --cuda-device-only -c }"; hcmd="${cmd/ -c / --cuda-host-only -fsyntax-only }"
      hcmd="${hcmd% -o *}"
      if (eval "$dcmd" && eval "$hcmd") > "$OUT/$arch/$name.log" 2>&1; then r=OK*; else r=FAIL; st=1; fi
    else r=FAIL; st=1; fi
    ncx=-
    if [ "$CONSTEXPR_PASS" = 1 ]; then
      ccmd="${cmd/ -c / --cuda-device-only -fsyntax-only $NVCC_LIKE }"; ccmd="${ccmd% -o *}"
      (eval "$ccmd") > "$OUT/$arch/$name.constexpr.raw.log" 2>&1
      grep -E ' error:' "$OUT/$arch/$name.constexpr.raw.log" | grep -vE '^(\.\./)?subprojects/|^/usr/|/subprojects/Kokkos/' \
        | grep -vE "'(move|forward)<" | sort -u > "$OUT/$arch/$name.constexpr.log"
      ncx=$(wc -l < "$OUT/$arch/$name.constexpr.log"); [ "$ncx" -gt 0 ] && { r=FAIL; st=1; }
    fi
    ne=$(grep -c ' error:' "$OUT/$arch/$name.log"); nw=$(grep -c ' warning:' "$OUT/$arch/$name.log")
    sp=$(grep -E 'bytes spill (stores|loads)' "$OUT/$arch/$name.log" | grep -vc ' 0 bytes spill stores, 0 bytes spill loads')
    mx=$(grep -oE '[0-9]+ bytes stack frame' "$OUT/$arch/$name.log" | awk '$1>m{m=$1}END{print m+0}')
    printf "%-6s %-26s %-4s errors=%-3s constexpr_errors=%-3s warnings=%-3s kernels_with_spills=%-3s max_stack=%sB\n" \
      "$arch" "$name" "$r" "$ne" "$ncx" "$nw" "$sp" "$mx"
  done
done
echo "(OK* = clang x86 host-backend crash; device codegen + host-side CUDA Sema checked instead)"
echo "CUDA COMPILE CHECK: $([ $st = 0 ] && echo PASS || echo FAIL)"; exit $st
