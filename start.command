#!/bin/bash
# macOS: double-click to launch 360 Views (reopens your last project; create or switch projects in the app).
exec "$(dirname "$0")/start.sh" "$@"
