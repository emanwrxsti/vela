#!/usr/bin/env bash
# SimpleMining (SMOS) custom-miner launcher for supr-meow-vela.
# SMOS runs custom miners from /home/miner/custom/<name>/ and exports the rig
# config into the environment. Set POOL_URL and WALLET in the SMOS miner config; the
# sd-turbo-int model downloads once and is cached under MODEL_DIR.
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/lib"

[[ -z $POOL_URL ]] && { echo "POOL_URL missing — set it in the SMOS miner config (e.g. vela.suprnova.cc:7601)"; exit 1; }
[[ -z $WALLET   ]] && { echo "WALLET missing — set your Vela wallet in the SMOS miner config"; exit 1; }

# ---- model: sd-turbo-int.poisd (~1.2 GB), fetched once and cached --------------------------------
MODEL_URL="${MODEL_URL:-https://vela.suprnova.cc/dl/sd-turbo-int.poisd}"
MODEL_SHA256="${MODEL_SHA256:-ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36}"
if [[ -z $MODEL_DIR ]]; then
  for d in /home/miner/models "$(pwd)/models"; do
    [[ -d $(dirname "$d") ]] && MODEL_DIR="$d" && break
  done
  MODEL_DIR="${MODEL_DIR:-$(pwd)/models}"
fi
if [[ -z $MODEL_PATH ]]; then
  mkdir -p "$MODEL_DIR"; MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
  if [[ ! -s $MODEL_PATH ]]; then
    echo "[supr-meow-vela] downloading model (~1.2 GB, one time): $MODEL_URL"
    curl -fL --retry 5 --no-progress-meter -o "$MODEL_PATH.part" "$MODEL_URL" || exit 1
    mv "$MODEL_PATH.part" "$MODEL_PATH"
  fi
  [[ -n $MODEL_SHA256 ]] && { echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || { echo "[supr-meow-vela] model checksum mismatch — refusing to mine"; rm -f "$MODEL_PATH"; exit 1; }; }
fi
[[ -s $MODEL_PATH ]] || { echo "[supr-meow-vela] no model file"; exit 1; }

args=( -o "$POOL_URL" -u "$WALLET" -p "${PASSWORD:-x}" --model "$MODEL_PATH" )
[[ -n $BATCH        ]] && args+=( --batch "$BATCH" )
[[ -n $LANES        ]] && args+=( --lanes "$LANES" )
[[ -n $DEVICES      ]] && args+=( --devices "$DEVICES" )
[[ -n $SUGGEST_DIFF ]] && args+=( --suggest-diff "$SUGGEST_DIFF" )
[[ ${TLS:-0} == 1   ]] && args+=( --tls )
[[ -n $EXTRA_ARGS   ]] && args+=( $EXTRA_ARGS )
exec ./supr-meow-vela "${args[@]}"
