# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items before continuing LOCAL_AGENT_TASK.md section 4.

## 2026-11-24 19:40 — review 1
**State:**
- 4.1, 4.2 (Batch A, all gates incl. the 2-rank fix 982c4ea) and 4.3 G5 are done.
- The user ran G7 on the V100: PASS (super_cat bitwise residuals; invwedge noise floor 1.00/1.05). Device-resident is ON.
- Kernel times per call: SetPrimVar 0.6 ms, Upwind 3.6 ms, Source 3.7 ms, Viscous 5.0 ms.
- Merged into su2kokkos: the script env-var paths (791070c).

**Finding:** rounds ~23–77 produced no code for 4.4 (LAX), only progress-log commits. The loop now treats log-only rounds as stuck.

**Agent: do next**
1. LAX (4.4): write code this session.
   - Smallest compilable step first: a `KOKKOS_INLINE_FUNCTION` port of the LAX flux (numerics/NEMO/convection/lax.cpp, ComputeResidual, without the Jacobian) in a new header next to CNEMOKokkosAUSM.hpp.
   - Build the serial target, then commit.
   - Next session: dispatch it in CNEMOKokkosUpwind.cpp (Centered_Residual path) behind the scheme check.
   - If 3 sessions pass without a compiling piece, mark 4.4-LAX BLOCKED and continue with 4.5.
2. Read at most 80 lines of any file at a time (lax.cpp is ~170 lines: read it in two halves).
