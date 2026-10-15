#!/usr/bin/env bash
#
# Launch a GPU container for this experiment and attach a shell.
#
#   ./start.sh          start (or attach to) the container
#   ./start.sh jupyter  start it running JupyterLab instead
#   ./start.sh stop     remove the container
#   ./start.sh logs     follow its setup logs
#
# The repo is bind-mounted, so edits on the host take effect immediately and
# nothing inside the container needs rebuilding after a code change.

set -euo pipefail

BASE_IMAGE="${AUTOFLIM_IMAGE:-gilson-base:latest}"
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
CONTAINER_NAME="autoflim_${EXP_NAME}"

HOST_HOME="${HOME}"
# Parent directory mounted as /workspace, so sibling datasets stay reachable.
HOST_WORKSPACE="${AUTOFLIM_WORKSPACE:-$(dirname "$EXP_DIR")}"
EXP_SUBDIR="${EXP_DIR#"${HOST_WORKSPACE}"/}"

container_exists() { docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; }
container_running() { docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; }

MODE="${1:-start}"

case "$MODE" in
  stop)
    echo "Stopping ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || echo "(not running)"
    exit 0
    ;;
  logs)
    docker logs -f "$CONTAINER_NAME"
    exit 0
    ;;
esac

if container_running; then
    echo "Attaching to running ${CONTAINER_NAME}..."
    exec docker exec -it "$CONTAINER_NAME" /bin/bash
fi

if container_exists; then
    echo "Removing stale ${CONTAINER_NAME}..."
    docker rm "$CONTAINER_NAME"
fi

ENTRYPOINT=$(cat <<'INNER'
# No `set -e`: we want a shell even if an install step fails.
cd "/workspace/$EXP_SUBDIR" || { echo "!! /workspace/$EXP_SUBDIR not found"; exec sleep infinity; }

echo "==> Installing pyflim (vendored)..."
pip install -e lib/pyflim --no-deps -q 2>&1 | tail -2

echo "==> Installing autoflim..."
pip install -e . -q 2>&1 | tail -2

echo "==> Environment ready."

if [ "${LAUNCH_MODE}" = "jupyter" ]; then
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
         --NotebookApp.token='' --notebook-dir="/workspace/$EXP_SUBDIR"
else
    exec sleep infinity
fi
INNER
)

echo "Experiment : $EXP_NAME"
echo "Container  : $CONTAINER_NAME"
echo "Workspace  : $HOST_WORKSPACE -> /workspace"

docker run \
    -u "$(id -u):$(id -g)" \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --ipc=host \
    --net=host \
    -d \
    -e LAUNCH_MODE="$MODE" \
    -e EXP_SUBDIR="$EXP_SUBDIR" \
    -v "${HOST_HOME}:/home/expuser" \
    -v "${HOST_WORKSPACE}:/workspace" \
    "$BASE_IMAGE" \
    /bin/bash -c "$ENTRYPOINT"

echo "Waiting for setup..."
timeout 180 sh -c '
    until docker logs "$1" 2>&1 | grep -q "Environment ready"; do sleep 1; done
' _ "$CONTAINER_NAME" && echo "Ready." || echo "Warning: setup timed out; attaching anyway."

exec docker exec -it "$CONTAINER_NAME" /bin/bash
