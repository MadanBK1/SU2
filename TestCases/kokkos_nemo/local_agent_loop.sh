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
MAX_TURNS=${MAX_TURNS:-40}                      # tool turns per session (bounds context use)
ROUND_TIMEOUT=${ROUND_TIMEOUT:-3h}              # hard wall-clock cap per session
MAX_ROUNDS=${MAX_ROUNDS:-200}
MAX_STUCK=${MAX_STUCK:-3}                       # stop after N rounds with no progress
PAUSE=${PAUSE:-20}
LOGS=$WS/agent_logs; mkdir -p "$LOGS"

# Point Claude Code at the local ollama server (Anthropic-compatible API).
export ANTHROPIC_BASE_URL="http://$OLLAMA_HOST" ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_API_KEY=""
export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16000}
export MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-4000}
export DISABLE_AUTOUPDATER=1

PROMPT="/no_think You are running unattended in a loop; each session starts with an empty context.
cd SU2. Follow LOCAL_AGENT_TASK.md exactly, starting with section 0 (git status/diff, read KOKKOS_PROGRESS.md).
Work only on branch $BRANCH; stay inside $WS.
Do ONE bounded chunk of the first unchecked item (an edit + build + check), keeping replies short.
Redirect long output to files and show only tails (cmd > log 2>&1; tail -30 log).
Before you finish this session ALWAYS: append status/commands/results/next step to KOKKOS_PROGRESS.md,
commit (WIP if gates not complete), and push to origin $BRANCH.
If every item in section 4 is done, write the line ALL ITEMS DONE at the end of KOKKOS_PROGRESS.md.
If you are blocked and need the human, write the line NEEDS HUMAN: <reason> at the end of KOKKOS_PROGRESS.md.
Then end the session."

state() { (cd "$REPO" && git rev-parse HEAD 2>/dev/null; md5sum KOKKOS_PROGRESS.md 2>/dev/null; git diff --stat | tail -1) | md5sum | cut -c1-12; }

command -v claude >/dev/null || { echo "claude CLI not found in PATH"; exit 1; }
cd "$REPO" && git checkout -q "$BRANCH" || { echo "cannot checkout $BRANCH"; exit 1; }
[ "$(git -C "$REPO" branch --show-current)" = "$BRANCH" ] || { echo "not on $BRANCH"; exit 1; }

stuck=0
for ((r=1; r<=MAX_ROUNDS; r++)); do
  [ -e "$WS/STOP_AGENT" ] && { echo "STOP_AGENT found; exiting."; break; }
  grep -q '^ALL ITEMS DONE' "$REPO/KOKKOS_PROGRESS.md" 2>/dev/null && { echo "ALL ITEMS DONE; exiting."; break; }
  grep -q '^NEEDS HUMAN' "$REPO/KOKKOS_PROGRESS.md" 2>/dev/null && { echo "NEEDS HUMAN: $(grep '^NEEDS HUMAN' "$REPO/KOKKOS_PROGRESS.md" | tail -1)"; break; }
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
  [ $stuck -ge $MAX_STUCK ] && { echo "No progress for $MAX_STUCK rounds; stopping (check $LOGS)."; \
       echo "NEEDS HUMAN: no progress for $MAX_STUCK rounds (see agent_logs)" >> "$REPO/KOKKOS_PROGRESS.md"; break; }
  sleep "$PAUSE"
done
echo "$(date) loop finished."
