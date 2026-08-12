#!/usr/bin/env bash
#
# Summarize a rotating training log (see README §17).
#
#   scripts/watch_rotation.sh                        # summary of the default log
#   scripts/watch_rotation.sh -l logs/other.log      # a different log
#   scripts/watch_rotation.sh -f                     # follow it live (Ctrl-C to stop)
#   scripts/watch_rotation.sh -n 20                  # show more history
#
# A rotating run logs one line per 100 steps for days, so the lines that
# actually tell you how it is going -- which chunk it is on, what the fixed
# validation set says, and whether the GPU is stalling on downloads -- get
# buried. This pulls just those out.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_ROOT/logs/mdcath_backbone_rotate.log"
LINES=8
FOLLOW=0

usage() {
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Options:
  -l, --log PATH   Log file to read (default: logs/mdcath_backbone_rotate.log).
  -n, --lines N    History lines per section (default: 8).
  -f, --follow     Stream the log instead of summarizing.
  -h, --help       Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--log)    LOG="${2:-}"; shift 2 ;;
    -n|--lines)  LINES="${2:-}"; shift 2 ;;
    -f|--follow) FOLLOW=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$LOG" ]]; then
  echo "error: no log at $LOG" >&2
  echo "       the run may not have started yet; check: tmux ls" >&2
  exit 3
fi

if [[ "$FOLLOW" -eq 1 ]]; then
  exec tail -f "$LOG"
fi

section() { printf '\n\033[1m%s\033[0m\n' "$1"; }

section "rotation progress  ($LOG)"
grep -E "cycle [0-9]+/[0-9]+, chunk" "$LOG" | tail -n "$LINES" || echo "  (no chunk started yet)"

section "training steps"
grep -E "^\S+ \S+ \| INFO \|.*step [0-9]+ epoch" "$LOG" | tail -n "$LINES" || echo "  (none yet)"

section "validation  (fixed holdout -- the only cross-chunk comparable signal)"
grep -E "val_loss=" "$LOG" | tail -n "$LINES" || echo "  (none yet)"

section "endpoint rollout"
grep -E "endpoint_rmsd:" "$LOG" | tail -n 3 || echo "  (none yet)"

section "download / disk"
grep -E "Waited [0-9]+s for chunk|shard\(s\) ready|Released chunk|Keeping chunk|GB free is below" "$LOG" \
  | tail -n "$LINES" || echo "  (none yet)"

section "problems"
problems="$(grep -E "ERROR|Non-finite|failed verification|Giving up|Skipping .*only .* GB free|Traceback" "$LOG" | tail -n "$LINES")"
if [[ -n "$problems" ]]; then echo "$problems"; else echo "  none"; fi

section "at a glance"
last_step="$(grep -oE "step [0-9]+ epoch" "$LOG" | tail -1 | grep -oE "[0-9]+")"
last_chunk="$(grep -E "cycle [0-9]+/[0-9]+, chunk" "$LOG" | tail -1 | sed -E 's/.*(cycle [0-9]+\/[0-9]+, chunk [0-9]+\/[0-9]+ \(cursor [0-9]+\)).*/\1/')"
best_val="$(grep -oE "val_loss=[0-9.]+" "$LOG" | cut -d= -f2 | sort -g | head -1)"
stall="$(grep -oE "cumulative download stall: [0-9]+s" "$LOG" | tail -1)"
echo "  step          : ${last_step:-0}"
echo "  position      : ${last_chunk:-not started}"
echo "  best val_loss : ${best_val:-n/a}"
echo "  ${stall:-cumulative download stall: 0s}"
echo "  log size      : $(du -h "$LOG" | cut -f1), last write $(date -r "$LOG" '+%Y-%m-%d %H:%M:%S')"
echo
