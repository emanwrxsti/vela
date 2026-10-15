#!/usr/bin/env bash
# Env-var front end for supr-meow-vela (Vela PoI v2 miner).
#
# REQUIRED
#   POOL_URL    host:port or stratum+tcp://host:port   (default vela.suprnova.cc:7601)
#   WALLET      vela1… payout address
#
# MODEL (defaults to the pinned sd-turbo-int model from the pool)
#   MODEL_PATH    path inside the container (use with a mounted volume)
#   MODEL_URL     fetched to $MODEL_DIR on first start, then cached
#   MODEL_SHA256  integrity check (default = the pinned model's hash)
#   MODEL_DIR     cache dir (default /models)
#
# OPTIONAL
#   WORKER, PASSWORD, DEVICES, BATCH, LANES, SUGGEST_DIFF, TLS=1, EXTRA_ARGS
#
# SHELL ACCESS (sshd starts only if one of these is set — handy on rented rigs)
#   SSH_PUBKEY    authorized_keys line (preferred)   SSH_PASSWORD  root password   SSH_PORT (22)
set -euo pipefail

POOL_URL="${POOL_URL:-vela.suprnova.cc:7601}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_URL="${MODEL_URL:-https://vela.suprnova.cc/dl/sd-turbo-int.poisd}"
MODEL_SHA256="${MODEL_SHA256:-ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36}"
SSHD_UP=0

die() {
  echo "error: $*" >&2
  if [[ $SSHD_UP == 1 ]]; then
    while true; do echo "[entrypoint] NOT MINING — $* (sshd up; log in and fix the env, then restart)"; sleep 300; done
  fi
  exit 2
}

start_sshd() {
  mkdir -p /root/.ssh /run/sshd; chmod 700 /root/.ssh; ssh-keygen -A >/dev/null 2>&1
  local pw_auth=no root_login=prohibit-password
  [[ -n "${SSH_PUBKEY:-}" ]]   && { echo "$SSH_PUBKEY" > /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; }
  [[ -n "${SSH_PASSWORD:-}" ]] && { echo "root:${SSH_PASSWORD}" | chpasswd; pw_auth=yes; root_login=yes; }
  mkdir -p /etc/ssh/sshd_config.d
  printf 'PasswordAuthentication %s\nPermitRootLogin %s\n' "$pw_auth" "$root_login" > /etc/ssh/sshd_config.d/zz-meow.conf
  if /usr/sbin/sshd -p "${SSH_PORT:-22}"; then SSHD_UP=1; echo "[entrypoint] sshd on ${SSH_PORT:-22}"; fi
  return 0
}

[[ -n "${SSH_PUBKEY:-}" || -n "${SSH_PASSWORD:-}" ]] && start_sshd

[[ -n "${WALLET:-}" ]] || die "WALLET is required (vela1… address)"
mkdir -p "$MODEL_DIR"

# ---- model: download once, verify, cache ----
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
  if [[ ! -s "$MODEL_PATH" ]]; then
    echo "[entrypoint] downloading model (~1.2 GB, one time): $MODEL_URL"
    curl -fL --retry 5 --retry-delay 5 -o "$MODEL_PATH.part" "$MODEL_URL" || die "model download failed"
    mv "$MODEL_PATH.part" "$MODEL_PATH"
  else
    echo "[entrypoint] using cached model: $MODEL_PATH"
  fi
fi
[[ -s "$MODEL_PATH" ]] || die "model not readable: $MODEL_PATH"
if [[ -n "${MODEL_SHA256:-}" ]]; then
  echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || die "model checksum mismatch — refusing to mine"
fi

USER_ARG="$WALLET"; [[ -n "${WORKER:-}" ]] && USER_ARG="${WALLET}.${WORKER}"
args=( -o "$POOL_URL" -u "$USER_ARG" -p "${PASSWORD:-x}" --model "$MODEL_PATH" )
[[ -n "${DEVICES:-}"      ]] && args+=( --devices "$DEVICES" )
[[ -n "${BATCH:-}"        ]] && args+=( --batch "$BATCH" )
[[ -n "${LANES:-}"        ]] && args+=( --lanes "$LANES" )
[[ -n "${SUGGEST_DIFF:-}" ]] && args+=( --suggest-diff "$SUGGEST_DIFF" )
[[ "${TLS:-0}" == "1"     ]] && args+=( --tls )
[[ -n "${EXTRA_ARGS:-}"   ]] && args+=( ${EXTRA_ARGS} )

echo "[entrypoint] pool=$POOL_URL user=$USER_ARG model=$(basename "$MODEL_PATH")"
exec /app/supr-meow-vela "${args[@]}"
