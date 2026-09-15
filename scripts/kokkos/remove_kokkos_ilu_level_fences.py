#!/usr/bin/env python3
"""Remove per-level Kokkos ILU fences while preserving level ordering.

The device ILU kernels are enqueued on one explicit DefaultExecutionSpace
instance. CUDA/HIP/SYCL queue ordering preserves forward/backward dependency
ordering between levels. A single fence is retained after the backward sweep,
before host/MPI communication can consume the result.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "Common/src/linear_algebra/CSysMatrixKokkos.cpp"
s = P.read_text()

marker = "KOKKOS ILU STREAM-ORDERED LEVELS"
if marker in s:
    print("Kokkos ILU stream-ordered level patch already present.")
    sys.exit(0)

old = '''  using execution_space = Kokkos::DefaultExecutionSpace;\n  using range_policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<unsigned long>>;\n'''
new = '''  using execution_space = Kokkos::DefaultExecutionSpace;\n  using range_policy = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<unsigned long>>;\n\n  /*--- KOKKOS ILU STREAM-ORDERED LEVELS\n   * Use one execution-space instance for the complete triangular solve.\n   * Successive kernels submitted to this instance are ordered on the same\n   * backend queue/stream, so level dependencies do not require a host fence\n   * after every kernel launch. ---*/\n  execution_space exec;\n'''
if s.count(old) != 1:
    sys.exit(f"ERROR: execution-space anchor count = {s.count(old)}")
s = s.replace(old, new, 1)

# Make both level kernels use the same explicit execution-space instance.
s, n = __import__('re').subn(
    r'range_policy\(begin, end\)',
    'range_policy(exec, begin, end)',
    s,
    count=2,
)
if n != 2:
    sys.exit(f"ERROR: expected 2 ILU range policies, replaced {n}")

for fence in [
    '    Kokkos::fence("SU2::Kokkos ILU forward level complete");\n',
    '    Kokkos::fence("SU2::Kokkos ILU backward level complete");\n',
]:
    if s.count(fence) != 1:
        sys.exit(f"ERROR: expected fence not found exactly once: {fence.strip()}")
    s = s.replace(fence, '', 1)

anchor = '''  /*--- Match the existing preconditioner semantics: communicate solved halo\n   * entries. Keep the result resident when the caller is a device Krylov solver. ---*/\n'''
replacement = '''  /*--- All forward/backward level kernels above are stream ordered. Synchronize\n   * once before MPI/host communication consumes the completed solve. ---*/\n  exec.fence("SU2::Kokkos ILU triangular solve complete");\n\n  /*--- Match the existing preconditioner semantics: communicate solved halo\n   * entries. Keep the result resident when the caller is a device Krylov solver. ---*/\n'''
if s.count(anchor) != 1:
    sys.exit(f"ERROR: communication anchor count = {s.count(anchor)}")
s = s.replace(anchor, replacement, 1)

P.write_text(s)
print(f"Patched {P}")
print("Removed per-level ILU fences; kernels now share one execution-space instance with one final fence.")
