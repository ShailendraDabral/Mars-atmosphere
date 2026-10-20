#!/usr/bin/env bash
# Open the interactive simulator locally at http://localhost:8000
cd "$(dirname "$0")/../docs" && python -m http.server 8000
