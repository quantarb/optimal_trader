#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_BIN="${CONDA_BIN:-/Users/johnnylee/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-optimal_trader}"
DJANGO_HOST="${DJANGO_HOST:-127.0.0.1}"
DJANGO_PORT="${DJANGO_PORT:-8000}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://${REDIS_HOST}:${REDIS_PORT}/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-rpc://}"

CELERY_PID=""
REDIS_PID=""

cleanup() {
  set +e
  if [[ -n "$CELERY_PID" ]]; then
    kill "$CELERY_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$REDIS_PID" ]]; then
    kill "$REDIS_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

redis_is_up() {
  python - "$REDIS_HOST" "$REDIS_PORT" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((host, port))
except Exception:
    print("0")
else:
    print("1")
finally:
    s.close()
PY
}

if [[ "$(redis_is_up)" != "1" ]]; then
  if command -v redis-server >/dev/null 2>&1; then
    echo "Starting local redis-server on ${REDIS_HOST}:${REDIS_PORT}"
    redis-server --bind "$REDIS_HOST" --port "$REDIS_PORT" --save "" --appendonly no >/tmp/optimal_trader_redis.log 2>&1 &
    REDIS_PID="$!"
    sleep 0.8
  elif command -v brew >/dev/null 2>&1; then
    echo "Starting Redis via Homebrew service"
    brew services start redis >/tmp/optimal_trader_redis.log 2>&1 || true
    sleep 1.2
  elif command -v docker >/dev/null 2>&1; then
    echo "Starting Redis via Docker container"
    docker rm -f optimal-trader-redis >/dev/null 2>&1 || true
    docker run -d --name optimal-trader-redis -p "${REDIS_PORT}:6379" redis:7 >/tmp/optimal_trader_redis.log 2>&1 || true
    sleep 1.2
  fi

  if [[ "$(redis_is_up)" != "1" ]]; then
    echo "Redis is still unavailable at ${REDIS_HOST}:${REDIS_PORT}."
    echo "Try one of:"
    echo "  1) brew install redis && brew services start redis"
    echo "  2) docker run -d --name optimal-trader-redis -p ${REDIS_PORT}:6379 redis:7"
    echo "  3) set CELERY_BROKER_URL to an existing Redis instance"
    exit 1
  fi

  if [[ -z "$REDIS_PID" ]]; then
    echo "Redis is reachable."
  else
    echo "Redis started with local process pid $REDIS_PID."
  fi
fi

echo "Starting Celery worker (env: ${CONDA_ENV})"
"$CONDA_BIN" run -n "$CONDA_ENV" celery -A celery_app:app worker -l info >/tmp/optimal_trader_celery.log 2>&1 &
CELERY_PID="$!"
sleep 1

echo "Starting Django at http://${DJANGO_HOST}:${DJANGO_PORT}"
exec "$CONDA_BIN" run -n "$CONDA_ENV" python manage.py runserver "${DJANGO_HOST}:${DJANGO_PORT}"
