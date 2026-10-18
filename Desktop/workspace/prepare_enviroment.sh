#!/bin/bash

packages=()
export DEBIAN_FRONTEND=noninteractive
if ! command -v python3 >/dev/null 2>&1; then
    packages+=(python3)
fi

if ! dpkg-query -W -f='${Status}' python3-venv 2>/dev/null | grep -qx 'install ok installed'; then
    packages+=(python3-venv)
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
    packages+=(python3-pip)
fi

if (( ${#packages[@]} > 0 )); then
    apt-get update
    apt-get install -y "${packages[@]}" --no-install-recommends
fi

python3 -m venv .venv 

source .venv/bin/activate

pip install -r requirements.txt
