#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shirizlw
"""

import subprocess
from os import path
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logging.info("")
    # DATA
    
    # path_src_train = '../../datasets/1-encoder/train/src-train.txt'
    path_src_train = path.join('train', 'src-train.txt')
    
    # path_tgt_train = '../../datasets/1-encoder/train/tgt-train.txt'
    path_tgt_train = path.join('train', 'tgt-train.txt')
    
    # path_src_val = '../../datasets/1-encoder/eval/src-val.txt'
    path_src_val = path.join('eval', 'src-val.txt')
    
    # path_tgt_val = '../../datasets/1-encoder/eval/tgt-val.txt'
    path_tgt_val = path.join('eval', 'src-val.txt')
    
    
    
    
    # BUILD VOCAB
    
    f = open('./build_vocab.sh', 'w')
    f.close()
    logging.info("Chmod 1")
    subprocess.run('chmod a+x build_vocab.sh', shell=True)
    
    f = open('build_vocab.sh', 'w')
    f.write('#!/usr/bin/env bash\n')
    f.write('onmt-build-vocab --size 50000 --save_vocab src-vocab.txt ' + path_src_train + '\n')
    f.write('onmt-build-vocab --size 50000 --save_vocab tgt-vocab.txt ' + path_tgt_train)
    f.close()
    
    print('Building vocabularies...')
    logging.info("build_vocab.h")
    subprocess.run('./build_vocab.sh')
    
    # TRAINING
    
    print('Starting Training...')
    f_sh = open('./train_model.sh', 'w')
    f_sh.close()
    logging.info("chmod2")
    subprocess.run('chmod a+x train_model.sh', shell=True)
    
    f_sh = open('./train_model.sh', 'w')
    f_sh.write('#!/usr/bin/env bash\n')
    f_sh.write('onmt-main --model Ablator.py --gpu_allow_growth --config data.yml --auto_config train --with_eval')
    f_sh.close()
    logging.info("train_model.sh")
    subprocess.run('./train_model.sh')
    

if __name__ == "__main__":
    main()


# python = "~3.11"
# pandas = "^1.5.3"
# requests = "^2.28.2"
# ndifflib = "^0.1.0"
# Levenshtein = "^0.20.9"
# OpenNMT-tf = "*"