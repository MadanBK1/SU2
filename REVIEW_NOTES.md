# Review notes for the local agent (written by the Claude reviewer; agent: read, never edit)

Newest section first. Do the **Agent: do next** items (tests/analysis), then write READY FOR REVIEW (see ROLE in LOCAL_AGENT_TASK.md).

## 2026-11-27 — review 34 (empty request; nothing merged; FIX: your memory files were too big for your context)

Reviewed: e01ea89d2e. Rounds 151-158 all ended rc=1 with no progress and the request had no reason line. Diagnosis:
KOKKOS_PROGRESS.md had grown to 161 KB and REVIEW_NOTES.md to 97 KB; reading them at session start filled your context
before any work. Fixed (history kept, nothing deleted): old log entries -> KOKKOS_PROGRESS_ARCHIVE.md, old review
sections -> REVIEW_NOTES_ARCHIVE.md. LOCAL_AGENT_TASK.md section 0: read `head -30` + `tail -40` of the progress file and
`head -45` of this file only; never read the archives whole (grep them). Keep progress entries ≤ 5 lines.
No new solver code on purpose: 4.6-Periodic (6dcd2bf) is still untested. Madan: nothing to run.

Agent: do next — the review-33 queue below (unchanged), starting NOW with step 1 (it takes 2 tool calls):
1. `pgrep -f per_queue.sh || setsid nohup bash TestCases/kokkos_nemo/per_queue.sh > /dev/null 2>&1 < /dev/null &`
   then `cat validation_logs/2026-11-27_4.6-Periodic/QUEUE_STATUS.txt`; append a 2-line progress entry, commit, end session.
2. Next sessions: steps 2-5 of review 33 (INDEX row + commit per END line; `git add -f` the logs; do not commit
   `.queue.lock`). Request `READY FOR REVIEW: MERGE: 4.6-Periodic — <results>` only when all steps PASS and are committed;
   `READY FOR REVIEW: HEAVY: 4.6-Periodic: <stage + iteration>` on a failure you cannot localize in 3 sessions.

## 2026-11-27 — review 33 (empty request again; nothing merged; validation runner for 4.6-Periodic)

Reviewed: 349539bd1b. Empty `reason:` again; rounds 143-150 made no progress (step 2 CPU sanity never ran to the end:
the runs die with the session). Nothing to merge. Dev this round (no solver code): `TestCases/kokkos_nemo/per_queue.sh`
— a DETACHED, resumable runner for review-31 steps 2-7 that writes every log directly into
validation_logs/2026-11-27_4.6-Periodic/ and one START/END line per step into QUEUE_STATUS.txt there (skips steps with a
`.<name>.done` marker; flock prevents double launch). check.sh: EXPECT keys for `invper_ausm_peroff` / `invper_ausm_pervenk`
(expect `periodic comms ...: host` + `resident ... OFF (`). Smoke-tested with stub binaries here; not run on SU2.

Agent: do next (each session is SHORT — do only these mechanical steps, no planning):
1. If `pgrep -f per_queue.sh` prints nothing and QUEUE_STATUS.txt has no `QUEUE FINISHED`/`STOP` line, launch it and end the session:
   `setsid nohup bash TestCases/kokkos_nemo/per_queue.sh > /dev/null 2>&1 < /dev/null &`
2. Every session: `cat validation_logs/2026-11-27_4.6-Periodic/QUEUE_STATUS.txt`. For each new END line add ONE INDEX row
   (gate = step name, result = the `::` summary, log = validation_logs/2026-11-27_4.6-Periodic/<name>.log), then
   `git add validation_logs && git commit -m "Validation: 4.6-Periodic <steps>"`. That counts as progress.
3. On `STOP: CPU sanity failed`: read the last 30 lines of cpu-per-r1-su2.log. If it is a periodic matching error, change
   2.0 -> -2.0 in meshes/invper_ausm.cfg, commit, `rm validation_logs/2026-11-27_4.6-Periodic/.cpu-per-r1.done`, relaunch (step 1).
   Any other error: `HEAVY: 4.6-Periodic: CPU sanity <first error line>`.
4. On a FAIL in g1-* (no `OVERALL PASS`, or `expected stage line missing`): wait for ab-per-r1 to finish, paste <= 20
   StageCompare lines with non-bitwise > 0 from ab-per-r1.log into the progress log, note the first differing iteration;
   if you cannot localize it in 3 sessions: `HEAVY: 4.6-Periodic: <stage + iteration>`.
5. When QUEUE FINISHED and g1/variants/ab/poison/regress are all OVERALL PASS and committed: `MERGE: 4.6-Periodic`
   (nf-per-r1 must be NOISEFLOOR PASS too). A request ALWAYS has a reason line; never file an empty one.

