#!/usr/bin/env bash
# HiveOS stats: parse the last "speed" line of the log. khs = inferences/s * 1000 (so HiveOS shows "inf/s" as kH/s-scaled numbers).
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
log="${CUSTOM_LOG_BASENAME}.log"
line=$(tail -n 200 "$log" 2>/dev/null | grep "speed " | tail -1)
total=$(echo "$line" | sed -n 's/.*speed \([0-9.]*\) inf\/s.*/\1/p'); total=${total:-0}
acc=$(echo "$line" | sed -n 's/.*shares A\([0-9]*\).*/\1/p'); rej=$(echo "$line" | sed -n 's/.*R\([0-9]*\) S.*/\1/p')
hs=(); temp=(); fan=()
while read -r g r; do hs+=("$r"); done < <(echo "$line" | grep -o "GPU[0-9]* [0-9.]* inf/s" | sed 's/GPU\([0-9]*\) \([0-9.]*\) inf\/s/\1 \2/')
[[ ${#hs[@]} -eq 0 ]] && hs=("$total")
for i in "${!hs[@]}"; do temp+=("$(nvidia-smi --id=$i --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)"); fan+=("$(nvidia-smi --id=$i --query-gpu=fan.speed --format=csv,noheader,nounits 2>/dev/null || echo 0)"); done
khs=$(awk -v t="$total" 'BEGIN{printf "%.3f", t*1000}')
stats=$(jq -nc --argjson hs "$(printf '%s\n' "${hs[@]}" | jq -R 'tonumber*1000' | jq -s .)" --argjson temp "$(printf '%s\n' "${temp[@]}" | jq -R 'tonumber' | jq -s .)" --argjson fan "$(printf '%s\n' "${fan[@]}" | jq -R 'tonumber' | jq -s .)" --arg ver "$CUSTOM_VERSION" --argjson acc "${acc:-0}" --argjson rej "${rej:-0}" '{hs:$hs, hs_units:"khs", temp:$temp, fan:$fan, uptime:0, ver:$ver, ar:[$acc,$rej], algo:"vela-poi"}')
echo "$stats"; echo "$khs"
