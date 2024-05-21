#!/usr/bin/env bash

FIELDWORK=${1:-"fieldwork"}

git lfs install
git clone --depth=1 -b src https://huggingface.co/datasets/wav2gloss/fieldwork ${FIELDWORK}

# untar everything
for f in ${FIELDWORK}/data/*/audio; do
    for split in train dev test; do
        if [ -f "${f}/${split}.tar.gz" ]; then
            tar -xvf "${f}/${split}.tar.gz" -C "${f}"
        fi
    done
done