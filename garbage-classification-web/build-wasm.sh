#!/usr/bin/env bash

rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &>/dev/null; then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
fi

# Set optimization flags
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis"

# Run wasm pack tool to build JS wrapper files and copy wasm to pkg directory.
mkdir -p classification-model/pkg
rm -rf classification-model/pkg/* || true
wasm-pack build ./classification-model --out-dir pkg --release --target web --no-default-features

# Manually copy the wasm to public directory
mkdir -p public/_nuxt
rm -f public/_nuxt/*.wasm || true
cp ./classification-model/pkg/*.wasm public/_nuxt/