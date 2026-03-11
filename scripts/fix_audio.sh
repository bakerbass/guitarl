#!/usr/bin/env bash
# fix_audio.sh — unstick the Scarlett USB audio device.
#
# Kills any process currently holding /dev/snd/*, then optionally reloads the
# snd_usb_audio kernel module so the device comes back clean.
#
# Usage:
#   ./scripts/fix_audio.sh           # just kill holders, no module reload
#   ./scripts/fix_audio.sh --reload  # also reload snd_usb_audio (needs sudo)

set -euo pipefail

echo "=== fix_audio: releasing /dev/snd/* ==="
sudo fuser -k /dev/snd/* 2>/dev/null && echo "  killed processes holding /dev/snd/*" \
                                      || echo "  no processes were holding /dev/snd/*"
sleep 0.5

if [[ "${1:-}" == "--reload" ]]; then
    echo "=== fix_audio: reloading snd_usb_audio ==="
    sudo rmmod snd_usb_audio 2>/dev/null || true
    sleep 1
    sudo modprobe snd_usb_audio
    sleep 2
    echo "  snd_usb_audio reloaded."
fi

echo "=== fix_audio: current ALSA capture devices ==="
arecord -l 2>/dev/null || echo "  (arecord not found)"

echo "=== fix_audio: done ==="
