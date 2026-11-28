# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-28 — review 44 (request 20261128T072358 had an EMPTY reason; nothing merged; stall fixed)
Reviewed: 550bcc800c (only loop autosaves; rounds 206-214 no progress). Older sections (41-43, Madan's
2026-11-28 priority order) moved verbatim to REVIEW_NOTES_ARCHIVE.md; KOKKOS_PROGRESS log rounds 148-214 moved
to KOKKOS_PROGRESS_ARCHIVE.md (tail -40 was 100% loop lines, so you saw no state). Priority order unchanged:
finish 4.6-PeriodicLim (limValid fix) -> zero-copy 2(a) bndsync -> 2(b) symmetry -> g7_all.sh. No new heavy code
this round: 2 dev items at HEAD are still untested.
Dev this round (scripts only, bash -n OK): TestCases/kokkos_nemo/bnd_queue.sh = review-42 B1-B5 as a detached,
resumable queue (builds, g1-bnd, g1-bndall, B3 history diff `HIST ... IDENTICAL`, B4 ledger, 13-case regress);
limfix_build_and_run.sh now runs bnd_queue.sh after the limfix queue finishes. local_agent_loop.sh LOOP HINT text
now says "do the first unfinished step / a running queue is not a stall" (applies after the loop restarts).
S below = BIN=<strict> REF_MODE=HOST RES_TOL=0 FIELD_TOL=0 (already inside the queues).

Agent: do next (ONE step per session; then commit and end the session)
1. `pgrep -af queue; tail -3 validation_logs/2026-11-28_4.6-PeriodicLim-fix/QUEUE_STATUS.txt validation_logs/2026-11-28_zc2a-bndsync/QUEUE_STATUS.txt`
   - nothing running and not both "QUEUE FINISHED" -> launch DETACHED and end the session:
     `setsid nohup bash TestCases/kokkos_nemo/limfix_build_and_run.sh > /dev/null 2>&1 < /dev/null &`
     (resumable: finished steps are skipped; build is a no-op if up to date)
   - running -> end the session. "BUILD FAILED" -> fix the compile error (smallest change), commit, delete the
     .build*.done markers in that dir, relaunch.
2. limfix dir has "QUEUE FINISHED": `git add -f validation_logs/2026-11-28_4.6-PeriodicLim-fix/` (*.log, BUILD_HEAD.txt,
   QUEUE_STATUS.txt), one validation_logs/INDEX.md row per step (copy the END line result), commit.
3. bndsync dir has "QUEUE FINISHED": same for validation_logs/2026-11-28_zc2a-bndsync/; paste the two ledger
   "sync boundary" byte lines (_bnd vs _bndall, from ledger-bnd-r1.log) into KOKKOS_PROGRESS.md; commit.
4. Any END line without "OVERALL: PASS" (or HIST not IDENTICAL): rerun that one case with SU2_KOKKOS_STAGE_VERIFY=1
   (S, same env/RUN_TAG) -> <name>-sv-r1.log in the same dir, paste <=20 StageCompare lines in KOKKOS_PROGRESS.md,
   commit, then request `HEAVY: <item>: <case> FAIL, see <log>`. A wrong EXPECT string = fix it yourself.
5. All green: last line of KOKKOS_PROGRESS.md
   `READY FOR REVIEW MERGE: 4.6-PeriodicLim + limValid fix + zc2a-bndsync — <one-line results>`
   (or `MERGE: 4.6-PeriodicLim + limValid fix — …` if only that is green). Never request with an empty reason.
Madan (optional, V100, after step 3): GPU timing of bndsync — CUDA build, axi_visccone RUN_TAG=_bnd and
_bndall (SU2_KOKKOS_NEMO_BND_ALL=1); archive both logs in validation_logs/2026-11-28_zc2a-bndsync/g7-bnd-v100-r1.log.
