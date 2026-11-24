#!/usr/bin/env bash
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
MAX_TURNS=${MAX_TURNS:-15}                      # tool turns per session (bounds context use)
ROUND_TIMEOUT=${ROUND_TIMEOUT:-3h}              # hard wall-clock cap per session
MAX_ROUNDS=${MAX_ROUNDS:-1000000}             # effectively unlimited
MAX_STUCK=${MAX_STUCK:-3}                       # after N no-progress rounds: back off + tell agent to change approach/skip
PAUSE=${PAUSE:-20}
LOGS=$WS/agent_logs; mkdir -p "$LOGS"

# Point Claude Code at the local ollama server (Anthropic-compatible API).
export ANTHROPIC_BASE_URL="http://$OLLAMA_HOST" ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_API_KEY=""
export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16000}
export MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-4000}
export DISABLE_AUTOUPDATER=1
# qwen via ollama is unrecognized -> Claude Code assumes a 1M window and never compacts.
# Cap auto-compact below the ollama context (OLLAMA_CONTEXT_LENGTH=65536). Bare number = tokens.
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-56000}

PROMPT="/no_think You are running unattended in a loop; each session starts with an empty context.
cd SU2. Follow LOCAL_AGENT_TASK.md exactly, starting with section 0 (git status/diff, read KOKKOS_PROGRESS.md).
Work only on branch $BRANCH; stay inside $WS.
Do ONE bounded chunk of the first unchecked item (an edit + build + check), keeping replies short.
Redirect long output to files and show only tails (cmd > log 2>&1; tail -30 log).
CONTEXT IS SMALL: never read a whole source file. Use grep -n to locate, then read at most 80 lines with offset/limit.
Never cat build logs or diffs in full (use tail -30, git diff --stat, or git diff -- <one file> | head -150).
Save progress to KOKKOS_PROGRESS.md and commit early (by your 10th tool call), then continue if turns remain.
Before you finish this session ALWAYS: append status/commands/results/next step to KOKKOS_PROGRESS.md,
commit (WIP if gates not complete), and push to origin $BRANCH.
If every item in section 4 is done, write the line ALL ITEMS DONE at the end of KOKKOS_PROGRESS.md.
Never wait for a human. If an item is blocked (credentials, permission refused, GPU busy, missing tool), mark it in
KOKKOS_PROGRESS.md as BLOCKED <item>: <reason> (retry later), and move on to the next unchecked item.
Retry BLOCKED items only after all other items are done or blocked.
If KOKKOS_PROGRESS.md contains a line starting with LOOP HINT:, obey the most recent one.
Then end the session."

state() { (cd "$REPO" && git rev-parse HEAD 2>/dev/null; md5sum KOKKOS_PROGRESS.md 2>/dev/null; git diff --stat | tail -1) | md5sum | cut -c1-12; }

command -v claude >/dev/null || { echo "claude CLI not found in PATH"; exit 1; }
cd "$REPO" && git checkout -q "$BRANCH" || { echo "cannot checkout $BRANCH"; exit 1; }
[ "$(git -C "$REPO" branch --show-current)" = "$BRANCH" ] || { echo "not on $BRANCH"; exit 1; }

stuck=0
for ((r=1; r<=MAX_ROUNDS; r++)); do
  [ -e "$WS/STOP_AGENT" ] && { echo "STOP_AGENT found; exiting."; break; }
  grep -q '^ALL ITEMS DONE' "$REPO/KOKKOS_PROGRESS.md" 2>/dev/null && { echo "ALL ITEMS DONE; exiting."; break; }
  # make sure ollama is up
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "$(date) ollama not reachable at $OLLAMA_HOST; retry in 60s"; sleep 60; continue; }

  before=$(state); log=$LOGS/round_$(date +%Y%m%d_%H%M%S)_$r.log; ln -sf "$log" "$LOGS/latest.log"
  echo "=== $(date) round $r (stuck=$stuck) -> $log"
  ( cd "$WS" && timeout "$ROUND_TIMEOUT" claude -p "$PROMPT" --model "$MODEL" --max-turns "$MAX_TURNS" \
        --permission-mode acceptEdits --output-format text ) > "$log" 2>&1
  rc=$?; echo "    exit=$rc  $(tail -c 300 "$log" | tr '\n' ' ')"

  # safety net: if the session died without saving, record it so the next round knows
  if [ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]; then
    echo "    WARNING: agent left $BRANCH; switching back"; git -C "$REPO" checkout -q "$BRANCH" || break
  fi
  after=$(state)
  if [ "$before" = "$after" ]; then stuck=$((stuck+1)); else stuck=0; fi
  if [ $rc -ne 0 ] || [ "$before" = "$after" ]; then
    printf '\n- %s [loop] round %d ended rc=%d with %s. Log: %s\n' "$(date -Is)" "$r" "$rc" \
      "$([ "$before" = "$after" ] && echo 'NO progress' || echo 'progress')" "$log" >> "$REPO/KOKKOS_PROGRESS.md"
  fi
  if [ $stuck -ge $MAX_STUCK ]; then
    echo "    no progress for $stuck rounds: backing off 10 min and steering the agent"
    printf '\nLOOP HINT: %s no progress for %d rounds on the current item. Try a different, smaller approach; if it still fails, mark it BLOCKED and move to the next unchecked item.\n' "$(date -Is)" "$stuck" >> "$REPO/KOKKOS_PROGRESS.md"
    sleep 600; stuck=0
  fi
  sleep "$PAUSE"
done
echo "$(date) loop finished."
