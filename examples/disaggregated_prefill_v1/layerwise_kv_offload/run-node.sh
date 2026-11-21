#!/usr/bin/env bash
# Execute on the designated A3 node, from its rendered node directory.
set -euo pipefail
cd "$(dirname "$0")"
action=${1:-help}
if [[ $action == help || $action == --help ]]; then
  echo 'Usage: run-node.sh create|master|engines|proxy|stop'
  echo 'create requires MODEL_DIR=/absolute/path/to/GLM-5.2-w4a8c8'
  exit 0
fi
readarray -t cfg < <(python3 - <<'PY'
import json
c=json.load(open('node.json'))
for k in ('node','profile','image','image_id'): print(c[k])
PY
)
node=${cfg[0]}; profile=${cfg[1]}; image=${cfg[2]}; expected_image=${cfg[3]}
name="glm52-layerwise-$profile"
label='glm52.layerwise.bundle'
case "$action" in
  create)
    : "${MODEL_DIR:?Set MODEL_DIR to the existing model checkpoint directory}"
    [[ $MODEL_DIR == /* && -f "$MODEL_DIR/config.json" ]]
    [[ $(docker image inspect --format '{{.Id}}' "$image") == "$expected_image" ]]
    # Never stop existing workloads to make room for this profile.
    npu-smi info | python3 -c '
import re,sys
text=sys.stdin.read()
pids=re.findall(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|",text,re.M)
if pids: raise SystemExit("NPUs are occupied; release the intended devices first: "+str(pids))
'
    mkdir -p logs
    docker run -dit --name "$name" --label "$label=$profile" \
      --network host --ipc private --shm-size 1t --privileged \
      --ulimit memlock=-1:-1 \
      -v "$PWD:/root/pd" -v "$MODEL_DIR:/models/GLM-5.2-w4a8c8:ro" \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
      -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
      -v /usr/local/dcmi:/usr/local/dcmi:ro \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
      "$image" bash
    docker exec -w / "$name" sha256sum -c /opt/layerwise-guide/SOURCE_SHA256SUMS > /dev/null
    docker exec -w / "$name" sha256sum -c /opt/layerwise-guide/PRESERVED_NATIVE_SHA256SUMS > /dev/null
    docker exec "$name" mount -o remount,huge=always /dev/shm
    ;;
  master|engines|proxy|stop)
    [[ $(docker inspect --format "{{index .Config.Labels \"$label\"}}" "$name") == "$profile" ]]
    if [[ $action == stop ]]; then
      docker stop --time 30 "$name"
    else
      [[ $action == engines || $node == p0 ]]
      docker exec -d "$name" bash "/root/pd/start_$action.sh"
    fi
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
