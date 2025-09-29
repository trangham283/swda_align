#!/bin/bash

# 1. Download MS-State corrected transcripts
# https://isip.piconepress.com/projects/switchboard/
wget https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
tar -xvzf switchboard_word_alignments.tar.gz

# 2. Download swda dataset and helper functions
git clone https://github.com/cgpotts/swda.git
cd swda
unzip swda.zip
