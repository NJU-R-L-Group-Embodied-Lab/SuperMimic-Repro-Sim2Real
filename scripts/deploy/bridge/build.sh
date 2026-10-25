#!/bin/bash
# Build g1_bridge + libg1bridge_client.so:  bash build.sh <unitree_sdk2 dir>   (x86_64 deploy PC or the aarch64 onboard computer PC2)
set -e
SDK=${1:?unitree_sdk2 dir}
ARCH=$(uname -m)
cd "$(dirname "$0")"
TL=$SDK/thirdparty/lib/$ARCH
g++ -O2 -std=c++17 -Wall -Wno-unused-parameter -I"$SDK/include" -I"$SDK/thirdparty/include" -I"$SDK/thirdparty/include/ddscxx" \
    g1_bridge.cpp -o g1_bridge -L"$SDK/lib/$ARCH" -lunitree_sdk2 -L"$TL" -lddscxx -lddsc -lpthread -lrt \
    -Wl,--disable-new-dtags,-rpath,"$TL"
g++ -O2 -std=c++17 -Wall -shared -fPIC g1_bridge_client.cpp -o libg1bridge_client.so -lrt
echo "built $(pwd)/g1_bridge $(pwd)/libg1bridge_client.so ($ARCH)"
