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
MODEL_URL="${MODEL_URL:-https://vela.suprnova.cc/dl/sd-turbo-int.poisd}"
MODEL_SHA256="${MODEL_SHA256:-ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36}"
# A custom URL may serve a different file; only add the built-in mirror for our
# default model. An explicit MODEL_FALLBACK_URL (including empty) overrides this.
if [[ ! ${MODEL_FALLBACK_URL+x} ]]; then
  MODEL_FALLBACK_URL=""
  [[ $MODEL_URL == https://vela.suprnova.cc/dl/sd-turbo-int.poisd ]] && \
    MODEL_FALLBACK_URL="https://www.suprnova.cc/dl/sd-turbo-int.poisd"
fi
if [[ -z $MODEL_PATH ]]; then
  mkdir -p "$MODEL_DIR" || exit 1
  MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
  if [[ ! -s $MODEL_PATH ]]; then
    urls=( "$MODEL_URL" )
    [[ -n $MODEL_FALLBACK_URL && $MODEL_FALLBACK_URL != "$MODEL_URL" ]] && urls+=( "$MODEL_FALLBACK_URL" )
    part="${MODEL_PATH}.part.$$"
    trap 'rm -f -- "$part"' EXIT
    for url in "${urls[@]}"; do
      echo "[supr-meow-vela] downloading model (1.2 GB, one time): $url"
      if curl -fL --retry 2 --retry-delay 2 --connect-timeout 15 \
          --speed-time 60 --speed-limit 1024 --no-progress-meter -o "$part" "$url" && \
          [[ -s $part ]] && echo "${MODEL_SHA256}  ${part}" | sha256sum -c -; then
        mv -- "$part" "$MODEL_PATH" || exit 1
        break
      fi
      echo "[supr-meow-vela] download failed or checksum mismatch: $url"
      rm -f -- "$part"
    done
    [[ -s $MODEL_PATH ]] || { echo "[supr-meow-vela] all model download sources failed"; exit 1; }
  fi
fi
[[ -s $MODEL_PATH ]] || { echo "[supr-meow-vela] no model — set MODEL_URL or MODEL_PATH in the flight sheet's extra config"; exit 1; }
echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || { echo "[supr-meow-vela] CHECKSUM MISMATCH — refusing to mine"; exit 1; }
args=( -o "$POOL_URL" -u "$WALLET" -p "${PASSWORD:-x}" --model "$MODEL_PATH" )
[[ -n $BATCH        ]] && args+=( --batch "$BATCH" )
[[ -n $LANES        ]] && args+=( --lanes "$LANES" )
[[ -n $DEVICES      ]] && args+=( --devices "$DEVICES" )
[[ -n $SUGGEST_DIFF ]] && args+=( --suggest-diff "$SUGGEST_DIFF" )
[[ ${TLS:-0} == 1   ]] && args+=( --tls )
[[ -n $EXTRA_ARGS   ]] && args+=( $EXTRA_ARGS )
exec ./supr-meow-vela "${args[@]}" 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
