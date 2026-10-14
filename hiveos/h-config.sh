#!/usr/bin/env bash
# Writes supr-meow-vela.conf from the flight sheet (sourced by h-run.sh).
[[ -z $CUSTOM_URL ]] && echo -e "${RED}CUSTOM_URL is empty${NOCOLOR}" && exit 1
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${RED}CUSTOM_TEMPLATE (wallet) is empty${NOCOLOR}" && exit 1
conf="POOL_URL=\"$CUSTOM_URL\"\nWALLET=\"$CUSTOM_TEMPLATE\"\nPASSWORD=\"${CUSTOM_PASS:-x}\"\n"
# Extra config lines are KEY=VALUE (MODEL_URL, MODEL_PATH, MODEL_SHA256, BATCH, LANES, DEVICES, SUGGEST_DIFF, TLS, EXTRA_ARGS)
[[ -n $CUSTOM_USER_CONFIG ]] && conf+="$CUSTOM_USER_CONFIG\n"
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME${NOCOLOR}" && exit 1
echo -e "$conf" > "$CUSTOM_CONFIG_FILENAME"
