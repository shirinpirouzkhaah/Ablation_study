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
    
    
    f = open('./build_vocab.sh', 'w')
    f.close()
    logging.info("Chmod 1")
    subprocess.run('chmod a+x build_vocab.sh', shell=True)
    
    f = open('build_vocab.sh', 'w')
    f.write('#!/usr/bin/env bash\n')
    f.write('python --version')
    f.close()
    
    print('Building vocabularies...')
    logging.info("build_vocab.h")
    subprocess.run('./build_vocab.sh')
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    