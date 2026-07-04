#!/usr/bin/env bash
set -Eeo pipefail
source "$(dirname "$0")/common.sh"
if [[ "${VIDEO_DEVICE:-AUTO}" != "AUTO" && -e "$VIDEO_DEVICE" ]]; then
  echo "$VIDEO_DEVICE"
  exit 0
fi
for dev in /dev/video*; do
  [[ -e "$dev" ]] || continue
  info="$(v4l2-ctl -d "$dev" --all 2>/dev/null || true)"
  formats="$(v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null || true)"
  if [[ "$info" == *RealSense* && "$formats" == *"'YUYV'"* && "$formats" == *"1280x720"* ]]; then
    if timeout 5 v4l2-ctl -d "$dev" \
      --set-fmt-video=width=1280,height=720,pixelformat=YUYV \
      --set-parm=15 --stream-mmap=3 --stream-count=3 --stream-to=/dev/null \
      >/dev/null 2>&1; then
      echo "$dev"
      exit 0
    fi
  fi
done
echo "[ERREUR] Flux couleur RealSense YUYV 1280x720 introuvable" >&2
exit 2
