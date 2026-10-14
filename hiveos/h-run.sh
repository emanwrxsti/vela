#!/usr/bin/env bash
# Launch supr-meow-vela. cwd is the package directory; HiveOS captures stdout.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
[[ -f $CUSTOM_CONFIG_FILENAME ]] && . "$CUSTOM_CONFIG_FILENAME"
# Bundled CUDA runtime + cuBLASLt live in lib/. The DRIVER (libcuda, libnvidia-ml) comes from HiveOS and must keep winning.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/lib"
[[ -z $POOL_URL ]] && echo "POOL_URL missing — set the pool in the flight sheet" && exit 1
[[ -z $WALLET   ]] && echo "WALLET missing — set the wallet template in the flight sheet" && exit 1
# ---- model: sd-turbo-int.poisd (1.2 GB), fetched once and cached ----------------------------------
MODEL_DIR="${MODEL_DIR:-/hive-config/models}"
if [[ -z $MODEL_PATH ]]; then
  if [[ -n $MODEL_URL ]]; then
    mkdir -p "$MODEL_DIR"; MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
    if [[ ! -s $MODEL_PATH ]]; then
      echo "[supr-meow-vela] downloading model (1.2 GB, one time): $MODEL_URL"
      curl -fL --retry 5 --no-progress-meter -o "$MODEL_PATH.part" "$MODEL_URL" || exit 1
      mv "$MODEL_PATH.part" "$MODEL_PATH"
    fi
    if [[ -n $MODEL_SHA256 ]]; then
      echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || { echo "[supr-meow-vela] CHECKSUM MISMATCH — refusing to mine"; rm -f "$MODEL_PATH"; exit 1; }
    fi
  else
    MODEL_PATH=$(find "$MODEL_DIR" . -maxdepth 1 -name '*.poisd' 2>/dev/null | head -1)
  fi
fi
[[ -s $MODEL_PATH ]] || { echo "[supr-meow-vela] no model — set MODEL_URL or MODEL_PATH in the flight sheet's extra config"; exit 1; }
args=( -o "$POOL_URL" -u "$WALLET" -p "${PASSWORD:-x}" --model "$MODEL_PATH" )
[[ -n $BATCH        ]] && args+=( --batch "$BATCH" )
[[ -n $LANES        ]] && args+=( --lanes "$LANES" )
[[ -n $DEVICES      ]] && args+=( --devices "$DEVICES" )
[[ -n $SUGGEST_DIFF ]] && args+=( --suggest-diff "$SUGGEST_DIFF" )
[[ ${TLS:-0} == 1   ]] && args+=( --tls )
[[ -n $EXTRA_ARGS   ]] && args+=( $EXTRA_ARGS )
exec ./supr-meow-vela "${args[@]}" 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
