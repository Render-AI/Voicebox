#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

. ./path.sh || exit 1;


# General configuration
stage=2              # Processes starts from the specified stage.
stop_stage=3     # Processes is stopped at the specified stage.

download_dir=downloads

feat_dir=data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Step 1: Download dataset and extract"
    mkdir -p ${download_dir}
    #wget --no-check-certificate https://www.openslr.org/resources/93/data_aishell3.tgz -O "${download_dir}/data_aishell3.tgz"
    echo "Extracting ..."
    mkdir -p ${download_dir}/aishell3
    tar -xf "${download_dir}/data_aishell3.tgz" -C "${download_dir}/aishell3"
    echo "Step 1: Done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Step 2: Prepare lexicon"
    python local/prepare_lexicon.py ${download_dir}/aishell3/phone_set.txt ${feat_dir}/lexicon.txt
    echo "Step 2: Done"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Step 3: Prepare transcript"
    python local/prepare_transcript.py --output_dir ${download_dir}/aishell3/train/wav/ ${download_dir}/aishell3/train/content.txt
    python local/prepare_transcript.py --output_dir ${download_dir}/aishell3/test/wav/ ${download_dir}/aishell3/test/content.txt
    echo "Step 3: Done"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Step 4: Download TextGrid"
    echo "Pls download AISHELL3.zip from https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4 , and put it to ${download_dir}"
    unzip ${download_dir}/AISHELL3.zip -d ${feat_dir}/
    echo "Step 4: Done"
fi

