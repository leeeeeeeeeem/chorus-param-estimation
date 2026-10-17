#!/bin/sh
# Reports whether the plugin's DSP has changed since crazy-chorus-dsp/ was frozen.
# Usage: scripts/check_dsp_drift.sh [path to the crazy-chorus repo, default ../crazy-chorus]
# Read-only: it changes nothing in either repository. See SPEC.md, "DSP".
cd "$(dirname "$0")/.." || exit 1
plugin="${1:-../crazy-chorus}"
if [ ! -d "$plugin/dsp" ]; then
    echo "no plugin DSP at $plugin/dsp; nothing to compare"
    exit 0
fi
if diff -r crazy-chorus-dsp "$plugin/dsp" >/dev/null; then
    echo "frozen copy matches $plugin/dsp"
else
    echo "crazy-chorus-dsp/ and $plugin/dsp differ:"
    diff -r crazy-chorus-dsp "$plugin/dsp"
    exit 1
fi
