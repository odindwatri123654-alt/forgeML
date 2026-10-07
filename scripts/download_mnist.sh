#!/usr/bin/env bash
# Скачивает MNIST в папку data/ (Linux / macOS / Git Bash).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
base="https://ossci-datasets.s3.amazonaws.com/mnist"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "data/$f" ]; then
        echo "data/$f уже есть"
        continue
    fi
    echo "скачиваю $f ..."
    curl -fsSL "$base/$f.gz" -o "data/$f.gz"
    gunzip -f "data/$f.gz"
done
echo "готово: MNIST лежит в data/"
