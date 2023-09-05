#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="1"

# General configuration
stage=0              # Processes starts from the specified stage.
stop_stage=0     # Processes is stopped at the specified stage.


download_dir=downloads

feat_dir=data



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Step 0: Get aligned transcript and phones"
    # This example will use already processed Aligned IPA transcriptions
    python local/textgrid2lab.py --store_phonesets "${feat_dir}/TextGrid" ${feat_dir}
    echo "Step 0: Done"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Step 1: Get melspec"
    echo "Step 1: Done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Step 2: Split train, eval, test"
    python local/split_dataset.py --unaligned ${feat_dir}/TextGrid/unaligned.txt ${download_dir}/aishell3 ${feat_dir}
    echo "Step 2: Done"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Step 3: Train audio model"
    python ../../voicebox/train.py fit --config config/config.yaml
    echo "Step 3: Done"
fi
