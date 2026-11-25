#!/usr/bin/env bash
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000
# Self-restarting loop for the local (ollama) Claude Code agent.
# Every round is a FRESH session (empty context): it reads LOCAL_AGENT_TASK.md +
# KOKKOS_PROGRESS.md, does one bounded chunk of work, saves progress, exits.
#
# Run from anywhere (inside tmux so it survives logout):
#   tmux new -s kkagent 'bash ~/software/AgenticTest/SU2/TestCases/kokkos_nemo/local_agent_loop.sh'
# Stop:    touch ~/software/AgenticTest/STOP_AGENT      (finishes current round, then exits)
# Watch:   tail -f ~/software/AgenticTest/agent_logs/latest.log
set -u
WS=${WS:-$HOME/software/AgenticTest}            # sandbox root (Claude is started here)
REPO=$WS/SU2
BRANCH=su2kokkos_qwenlocal_agent
MODEL=${MODEL:-qwen3.8:27b}
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11435}
MAX_TURNS=${MAX_TURNS:-25}                      # tool turns per session (bounds context use)
ROUND_TIMEOUT=${ROUND_TIMEOUT:-3h}              # hard wall-clock cap per session
MAX_ROUNDS=${MAX_ROUNDS:-1000000}             # effectively unlimited
MAX_STUCK=${MAX_STUCK:-3}                       # after N no-progress rounds: back off + tell agent to change approach/skip
PAUSE=${PAUSE:-20}
LOGS=$WS/agent_logs; mkdir -p "$LOGS"
PUSH_URL=${PUSH_URL:-git@github.com:MadanBK1/SU2.git}   # SSH (key on roam148-045); fetch stays on https origin
REVIEW_EVERY_ROUNDS=${REVIEW_EVERY_ROUNDS:-8}           # request a review at least every N rounds of real work
REVIEW_POLL=${REVIEW_POLL:-300}                         # seconds between checks for the reviewer's answer
REVIEW_MAX_WAIT=${REVIEW_MAX_WAIT:-21600}               # give up waiting after 6 h and continue
rounds_since_review=0

# Point Claude Code at the local ollama server (Anthropic-compatible API).
export ANTHROPIC_BASE_URL="http://$OLLAMA_HOST" ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_API_KEY=""
export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16000}
export MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-4000}
export DISABLE_AUTOUPDATER=1
# Claude Code's auto-compact window has a 100k minimum, above the 64k ollama context, so it
# cannot help here; MAX_TURNS per round is what bounds context. Leave it unset.
unset CLAUDE_CODE_AUTO_COMPACT_WINDOW
# Never block on a credential prompt in unattended mode: a push without auth fails fast instead.
export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true SSH_ASKPASS=/bin/true GIT_SSH_COMMAND="ssh -o BatchMode=yes"

PROMPT="/no_think You are running unattended in a loop; each session starts with an empty context.
cd SU2. Follow LOCAL_AGENT_TASK.md exactly, starting with section 0 (git status/diff, read KOKKOS_PROGRESS.md).
Work only on branch $BRANCH; stay inside $WS.
Do ONE bounded chunk of the first unchecked item (an edit + build + check), keeping replies short.
Redirect long output to files and show only tails (cmd > log 2>&1; tail -30 log).
CONTEXT IS SMALL: never read a whole source file. Use grep -n to locate, then read at most 80 lines with offset/limit.
Never cat build logs or diffs in full (use tail -30, git diff --stat, or git diff -- <one file> | head -150).
Save progress to KOKKOS_PROGRESS.md and commit early (by your 10th tool call), then continue if turns remain.
Before you finish this session ALWAYS: append status/commands/results/next step to KOKKOS_PROGRESS.md,
commit LOCALLY (WIP if gates not complete). Do NOT run git push.
If every item in section 4 is done, write the line ALL ITEMS DONE at the end of KOKKOS_PROGRESS.md.
Never wait for a human. If an item is blocked (credentials, permission refused, GPU busy, missing tool), mark it in
KOKKOS_PROGRESS.md as BLOCKED <item>: <reason> (retry later), and move on to the next unchecked item.
Retry BLOCKED items only after all other items are done or blocked.
If KOKKOS_PROGRESS.md contains a line starting with LOOP HINT:, obey the most recent one.
REVIEW_NOTES.md (written by the Claude reviewer every ~2 h; never edit it yourself) has priority over your own plan:
read its newest section first and do its Agent-do-next items before continuing the task list.
When you finish a work item or a meaningful compiled+tested chunk, write the line READY FOR REVIEW: <what> as the
last line of KOKKOS_PROGRESS.md and end the session; the loop then pushes and waits for the Claude reviewer.
Standing goals: improve GPU speedup AND keep numerical accuracy (strict gate G1 bitwise, noise floor <= 10);
never trade accuracy for speed.
Then end the session."

# "Progress" = change to anything except the progress log (log-only rounds count as stuck).
state() { (cd "$REPO" && git log -1 --format=%H -- . ':!KOKKOS_PROGRESS.md'; git diff --stat -- . ':!KOKKOS_PROGRESS.md' | tail -1) | md5sum | cut -c1-12; }

command -v claude >/dev/null || { echo "claude CLI not found in PATH"; exit 1; }
cd "$REPO" && git checkout -q "$BRANCH" || { echo "cannot checkout $BRANCH"; exit 1; }
[ "$(git -C "$REPO" branch --show-current)" = "$BRANCH" ] || { echo "not on $BRANCH"; exit 1; }

stuck=0
for ((r=1; r<=MAX_ROUNDS; r++)); do
  [ -e "$WS/STOP_AGENT" ] && { echo "STOP_AGENT found; exiting."; break; }
  grep -q '^ALL ITEMS DONE' "$REPO/KOKKOS_PROGRESS.md" 2>/dev/null && { echo "ALL ITEMS DONE; exiting."; break; }
  # make sure ollama is up
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "$(date) ollama not reachable at $OLLAMA_HOST; retry in 60s"; sleep 60; continue; }

  # pull reviewer notes (REVIEW_NOTES.md) and reviewer fixes from GitHub before the round
  if timeout 120 git -C "$REPO" fetch -q origin "$BRANCH" </dev/null 2>/dev/null; then
    if ! git -C "$REPO" merge -q --no-edit "origin/$BRANCH" </dev/null >/dev/null 2>&1; then
      git -C "$REPO" merge --abort 2>/dev/null; echo "    merge of origin/$BRANCH conflicted; skipped this round"
    fi
  fi
  before=$(state); log=$LOGS/round_$(date +%Y%m%d_%H%M%S)_$r.log; ln -sf "$log" "$LOGS/latest.log"
  echo "=== $(date) round $r (stuck=$stuck) -> $log"
  ( cd "$WS" && timeout "$ROUND_TIMEOUT" claude -p "$PROMPT" --model "$MODEL" --max-turns "$MAX_TURNS" \
        --permission-mode acceptEdits --output-format text ) > "$log" 2>&1
  rc=$?; echo "    exit=$rc  $(tail -c 300 "$log" | tr '\n' ' ')"

  # safety net: if the session died without saving, record it so the next round knows
  if [ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]; then
    echo "    WARNING: agent left $BRANCH; switching back"; git -C "$REPO" checkout -q "$BRANCH" || break
  fi
  # autosave: commit (locally) whatever the round left uncommitted (tracked edits + progress file only)
  if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || \
     [ -n "$(git -C "$REPO" status --porcelain -- KOKKOS_PROGRESS.md)" ]; then
    git -C "$REPO" add -u && git -C "$REPO" add KOKKOS_PROGRESS.md 2>/dev/null
    git -C "$REPO" reset -q -- subprojects/CoolProp subprojects/.wraplock 2>/dev/null
    git -C "$REPO" commit -q -m "WIP loop autosave round $r ($(date -Is)): uncommitted work left by the agent session; gates not verified" \
      && echo "    autosaved round $r"
  fi
  # ---- review handshake: push, then wait for the reviewer's answer (REVIEW_DONE == REVIEW_REQUEST id) ----
  rounds_since_review=$((rounds_since_review+1))
  if tail -3 "$REPO/KOKKOS_PROGRESS.md" 2>/dev/null | grep -q '^READY FOR REVIEW' || [ $rounds_since_review -ge "$REVIEW_EVERY_ROUNDS" ]; then
    rid="$(date +%Y%m%dT%H%M%S)-$(git -C "$REPO" rev-parse --short HEAD)"
    printf 'id: %s\nhead: %s\nreason: %s\n' "$rid" "$(git -C "$REPO" rev-parse HEAD)" \
      "$(tail -3 "$REPO/KOKKOS_PROGRESS.md" | grep '^READY FOR REVIEW' | tail -1 || echo "periodic ($rounds_since_review rounds)")" > "$REPO/REVIEW_REQUEST"
    git -C "$REPO" add REVIEW_REQUEST && git -C "$REPO" commit -q -m "REVIEW REQUEST $rid"
    if timeout 180 git -C "$REPO" push -q "$PUSH_URL" "$BRANCH" </dev/null >/dev/null 2>&1; then
      echo "    review requested ($rid); waiting for Claude (poll ${REVIEW_POLL}s, max ${REVIEW_MAX_WAIT}s)"
      waited=0
      while [ $waited -lt "$REVIEW_MAX_WAIT" ] && [ ! -e "$WS/STOP_AGENT" ]; do
        sleep "$REVIEW_POLL"; waited=$((waited+REVIEW_POLL))
        timeout 120 git -C "$REPO" fetch -q origin "$BRANCH" </dev/null 2>/dev/null || continue
        if git -C "$REPO" show "origin/$BRANCH:REVIEW_DONE" 2>/dev/null | grep -q "^id: $rid"; then
          git -C "$REPO" merge -q --no-edit "origin/$BRANCH" </dev/null >/dev/null 2>&1 || git -C "$REPO" merge --abort 2>/dev/null
          echo "    review received after ${waited}s; continuing"; break
        fi
      done
      [ $waited -ge "$REVIEW_MAX_WAIT" ] && echo "    no review after ${REVIEW_MAX_WAIT}s; continuing without it"
    else
      echo "    push failed; continuing without review this time"
    fi
    rounds_since_review=0
  fi
  after=$(state)
  if [ "$before" = "$after" ]; then stuck=$((stuck+1)); else stuck=0; fi
  if [ $rc -ne 0 ] || [ "$before" = "$after" ]; then
    printf '\n- %s [loop] round %d ended rc=%d with %s. Log: %s\n' "$(date -Is)" "$r" "$rc" \
      "$([ "$before" = "$after" ] && echo 'NO progress' || echo 'progress')" "$log" >> "$REPO/KOKKOS_PROGRESS.md"
  fi
  if [ $stuck -ge $MAX_STUCK ]; then
    echo "    no progress for $stuck rounds: backing off 10 min and steering the agent"
    printf '\nLOOP HINT: %s no progress for %d rounds on the current item. No code changed (only the progress log). Stop planning: in THIS session write code immediately - the smallest compilable piece (e.g. one device function + its call site), build it, commit it. If you cannot, mark the item BLOCKED and move to the next unchecked item.\n' "$(date -Is)" "$stuck" >> "$REPO/KOKKOS_PROGRESS.md"
    sleep 600; stuck=0
  fi
  sleep "$PAUSE"
done
echo "$(date) loop finished."
