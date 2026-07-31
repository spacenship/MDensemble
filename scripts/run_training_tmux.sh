#!/usr/bin/env bash
#
# Launch a training run inside a detached tmux session so it survives SSH
# disconnects, and so the shell you started it from stays usable.
#
#   scripts/run_training_tmux.sh --config configs/mdcath_full.yaml
#   scripts/run_training_tmux.sh -c configs/mdcath_large.yaml -s large -g 1
#   scripts/run_training_tmux.sh -c configs/mdcath_full.yaml -g 0,1 -n 2   # 2-GPU DDP
#
# Runs with the af3 conda environment by default (override: RUN_PYTHON=...).
#
# Then:
#   tmux attach -t <session>        # watch it live  (detach again: Ctrl-b d)
#   tail -f <log>                   # or just follow the log
#   tmux kill-session -t <session>  # stop it
#
# The run refuses to start if the session name, log file, or checkpoint
# directory already exists, so an accidental re-launch can never silently
# overwrite a run in progress. Pass --force to override.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to the af3 conda environment, which is the one with the full
# dependency set (torch + h5py + transformers). Note that a bare `python3`
# is not a safe default here: in a fresh login shell it resolves to the base
# conda env, which has torch but no h5py, so an mdCATH run would fail
# minutes in. Override with RUN_PYTHON=... if you use a different env.
DEFAULT_PYTHON="/opt/conda/envs/af3/bin/python3"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="$(command -v python3)"
RUN_PYTHON="${RUN_PYTHON:-$DEFAULT_PYTHON}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$REPO_ROOT/logs}"

CONFIG=""
SESSION=""
GPU="0"
LOG=""
FORCE=0
NPROC="1"
MASTER_PORT=

usage() {
  # Print the leading comment block only (stop at the first non-comment line),
  # so the header doc and --help never drift apart.
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Options:
  -c, --config PATH    Config YAML to train with (required).
  -s, --session NAME   tmux session name (default: derived from the config name).
  -g, --gpu LIST       CUDA_VISIBLE_DEVICES value (default: 0). For multi-GPU
                       pass every device, e.g. -g 0,1.
  -n, --nproc N        Processes for DistributedDataParallel via torchrun
                       (default: 1 = plain single-process run). Must match the
                       number of GPUs listed in --gpu.
      --master-port P  torchrun rendezvous port (default: a free port).
  -l, --log PATH       Log file path (default: $RUN_LOG_DIR/<session>.log).
      --force          Allow reusing an existing log / checkpoint directory.
  -h, --help           Show this message.

Environment:
  RUN_PYTHON   Python interpreter (default: the af3 conda env).
  RUN_LOG_DIR  Directory for logs (default: <repo>/logs).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)  CONFIG="${2:-}"; shift 2 ;;
    -s|--session) SESSION="${2:-}"; shift 2 ;;
    -g|--gpu)     GPU="${2:-}"; shift 2 ;;
    -n|--nproc)   NPROC="${2:-}"; shift 2 ;;
    --master-port) MASTER_PORT="${2:-}"; shift 2 ;;
    -l|--log)     LOG="${2:-}"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "error: --config is required" >&2
  usage >&2
  exit 2
fi

command -v tmux >/dev/null 2>&1 || { echo "error: tmux is not installed" >&2; exit 3; }
[[ -f "$REPO_ROOT/$CONFIG" || -f "$CONFIG" ]] || { echo "error: config not found: $CONFIG" >&2; exit 3; }
[[ -x "$RUN_PYTHON" ]] || { echo "error: RUN_PYTHON is not executable: $RUN_PYTHON" >&2; exit 3; }

if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [[ "$NPROC" -lt 1 ]]; then
  echo "error: --nproc must be a positive integer, got: $NPROC" >&2
  exit 2
fi

# A mismatch here is a classic way to silently train on the wrong number of
# GPUs (or deadlock), so check it up front.
GPU_COUNT="$(awk -F, '{print NF}' <<<"$GPU")"
if [[ "$NPROC" -gt "$GPU_COUNT" ]]; then
  echo "error: --nproc $NPROC exceeds the $GPU_COUNT device(s) in --gpu '$GPU'" >&2
  echo "       e.g. for 2-GPU DDP use: --gpu 0,1 --nproc 2" >&2
  exit 2
fi

# Default the session name to the config's basename (mdcath_full.yaml -> mdcath_full).
if [[ -z "$SESSION" ]]; then
  SESSION="$(basename "$CONFIG")"
  SESSION="${SESSION%.yaml}"
fi
[[ -z "$LOG" ]] && LOG="$RUN_LOG_DIR/${SESSION}.log"
mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "error: tmux session '$SESSION' already exists (attach with: tmux attach -t $SESSION)" >&2
  exit 4
fi

# Preflight: read the checkpoint dir back out of the config (rather than
# guessing it -- this is what a careless re-launch would overwrite) and check
# that the chosen interpreter can actually import what this config needs.
#
# The dependency check matters because `python3` resolves differently
# depending on how the shell was started: a fresh login shell here picks the
# base conda env, which has torch but *not* h5py, so an mdCATH run would die
# minutes in. Better to fail immediately with an actionable message.
PREFLIGHT="$("$RUN_PYTHON" - "$REPO_ROOT/$CONFIG" <<'PY' 2>&1
import importlib
import sys

try:
    import yaml
except ImportError:
    print("ERROR|this interpreter has no PyYAML")
    sys.exit(0)

with open(sys.argv[1]) as fh:
    config = yaml.safe_load(fh) or {}

data = config.get("data") or {}
model = config.get("model") or {}

required = {"torch": "torch", "yaml": "PyYAML", "numpy": "numpy"}
if data.get("source") == "mdcath":
    required["h5py"] = "h5py (needed for data.source: mdcath)"
if ((model.get("esm") or {}).get("enabled")):
    required["transformers"] = "transformers (needed for model.esm.enabled)"

missing = []
for module, label in required.items():
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(label)

if missing:
    print("ERROR|missing: " + ", ".join(missing))
else:
    print("OK|" + str((config.get("train") or {}).get("ckpt_dir", "")))
PY
)"

if [[ "$PREFLIGHT" == ERROR\|* || "$PREFLIGHT" != OK\|* ]]; then
  echo "error: $RUN_PYTHON cannot run this config" >&2
  echo "       ${PREFLIGHT#*|}" >&2
  echo "       set a different interpreter, e.g.:" >&2
  echo "         RUN_PYTHON=/opt/conda/envs/af3/bin/python3 $0 --config $CONFIG" >&2
  exit 6
fi
CKPT_DIR="${PREFLIGHT#OK|}"

if [[ "$FORCE" -eq 0 ]]; then
  for existing in "$LOG" "${CKPT_DIR:+$REPO_ROOT/$CKPT_DIR}"; do
    [[ -n "$existing" && -e "$existing" ]] || continue
    echo "error: refusing to overwrite existing output: $existing" >&2
    echo "       pass --force to proceed anyway, or choose another --session." >&2
    exit 5
  done
fi

# -u keeps Python's output unbuffered so `tail -f` shows progress live rather
# than in 8 KB bursts. `exec bash` keeps the pane open after the run ends so
# the exit status stays visible when attaching.
if [[ "$NPROC" -gt 1 ]]; then
  # Pick a free rendezvous port unless one was given, so concurrent DDP runs
  # on the same host do not collide.
  if [[ -z "$MASTER_PORT" ]]; then
    MASTER_PORT="$("$RUN_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
  fi
  LAUNCHER="$(printf '%q' "$(dirname "$RUN_PYTHON")/torchrun") --nproc_per_node=$(printf '%q' "$NPROC") --master_port=$(printf '%q' "$MASTER_PORT")"
else
  LAUNCHER="$(printf '%q' "$RUN_PYTHON") -u"
fi

COMMAND="cd $(printf '%q' "$REPO_ROOT") && \
CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPU") $LAUNCHER train.py \
--config $(printf '%q' "$CONFIG") 2>&1 | tee $(printf '%q' "$LOG"); \
echo \"[exit \${PIPESTATUS[0]}] finished \$(date -Is)\" | tee -a $(printf '%q' "$LOG"); exec bash"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$COMMAND"

cat <<EOF
started tmux session : $SESSION
  config             : $CONFIG
  GPU                : $GPU
  processes          : $NPROC$( [[ "$NPROC" -gt 1 ]] && echo "  (DDP via torchrun, port $MASTER_PORT)" )
  log                : $LOG
  checkpoints        : ${CKPT_DIR:-<not set in config>}

  watch live         : tmux attach -t $SESSION      (detach: Ctrl-b then d)
  follow log         : tail -f $LOG
  list sessions      : tmux ls
  stop               : tmux kill-session -t $SESSION
EOF
