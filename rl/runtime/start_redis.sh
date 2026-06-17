#!/usr/bin/env bash
# Ghost-league Redis for RLlib training.
#
# training_data layout (see league_initialization/paths.py):
#   redis-server/  Redis AOF/RDB bind mount (in-memory league keys)
#   redis/         Lean league safetensors exports (hydrate when Redis is empty)
#   RLlib/         RLlib Tune checkpoints (separate track; not stored in Redis)
#
# Usage:
#   ./start_redis.sh          start Redis (idempotent)
#   ./start_redis.sh status   container + connectivity + league seed hint
#   ./start_redis.sh stop     stop the container
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIEPCUSTOM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_DATA="$DIEPCUSTOM_ROOT/training_data"

REDIS_SERVER_DATA="${REDIS_DATA_DIR:-$TRAINING_DATA/redis-server}"
LEAGUE_EXPORT_DIR="$TRAINING_DATA/redis"
RLLIB_CHECKPOINT_DIR="$TRAINING_DATA/RLlib"

NAME="${REDIS_CONTAINER_NAME:-rllib-redis}"
PORT="${REDIS_PORT:-6379}"
IMAGE="${REDIS_IMAGE:-redis:7-alpine}"

CMD="${1:-start}"

redis_exec() {
  docker exec "$NAME" redis-cli "$@"
}

wait_for_redis() {
  local attempt
  for attempt in $(seq 1 30); do
    if redis_exec ping 2>/dev/null | grep -qx PONG; then
      return 0
    fi
    sleep 0.2
  done
  echo "Redis did not respond to PING on localhost:$PORT" >&2
  return 1
}

ensure_training_dirs() {
  mkdir -p "$REDIS_SERVER_DATA" "$LEAGUE_EXPORT_DIR" "$RLLIB_CHECKPOINT_DIR"
}

start_redis() {
  ensure_training_dirs

  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "$NAME already running on localhost:$PORT"
    wait_for_redis
    print_status
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME" >/dev/null
  else
    docker run -d \
      --name "$NAME" \
      -p "$PORT:6379" \
      -v "$REDIS_SERVER_DATA:/data" \
      "$IMAGE" \
      redis-server --appendonly yes --save "" >/dev/null
  fi

  wait_for_redis
  echo "Redis container $NAME is running on localhost:$PORT"
  print_status
}

stop_redis() {
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    docker stop "$NAME" >/dev/null
    echo "Stopped $NAME"
  else
    echo "$NAME is not running"
  fi
}

league_export_files_present() {
  find "$LEAGUE_EXPORT_DIR" \( -name 'iter_*.safetensors' -o -name 'iter_*.safetensors.zst' \) -print -quit 2>/dev/null | grep -q .
}

print_league_hint() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "League: start Redis first, then seed once:"
    echo "  python rl/runtime/train.py seed"
    return
  fi

  if redis_exec EXISTS policy:A:0 2>/dev/null | grep -qx 1; then
    local key_count
    key_count="$(redis_exec --scan --pattern 'policy:*' 2>/dev/null | wc -l | tr -d ' ')"
    echo "League: seeded in Redis ($key_count policy:* keys)"
  elif league_export_files_present; then
    echo "League: Redis empty, but SSD exports exist under $LEAGUE_EXPORT_DIR"
    echo "  Training will hydrate Redis on init (LeagueBootstrapCallback)"
  else
    echo "League: not seeded. Run once before first training:"
    echo "  cd $SCRIPT_DIR"
    echo "  python rl/runtime/train.py seed"
  fi
}

print_status() {
  echo "Paths:"
  echo "  redis-server (AOF-only, no RDB): $REDIS_SERVER_DATA"
  echo "  redis (exports):                 $LEAGUE_EXPORT_DIR"
  echo "  RLlib (checkpoints):             $RLLIB_CHECKPOINT_DIR"
  print_league_hint
}

case "$CMD" in
  start)
    start_redis
    ;;
  stop)
    stop_redis
    ;;
  status)
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      wait_for_redis
      echo "$NAME running on localhost:$PORT"
    else
      echo "$NAME not running"
    fi
    print_status
    ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac
