#!/usr/bin/env bash
onmt-build-vocab --size 50000 --save_vocab src-vocab.txt train/src-train.txt
onmt-build-vocab --size 50000 --save_vocab tgt-vocab.txt train/tgt-train.txt