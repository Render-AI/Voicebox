#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="1"

# General configuration
stage=3              # Processes starts from the specified stage.
stop_stage=5     # Processes is stopped at the specified stage.


download_dir=downloads

feat_dir=data



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Step 0: Get aligned transcript and phones"
    # This example will use already processed Aligned IPA transcriptions
    python local/textgrid2lab.py "${feat_dir}/TextGrid" ${feat_dir}
    echo "Step 0: Done"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Step 1: Get melspectrogram"
    # This example will use 
    python local/melspec.py "${download_dir}/LJSpeech-1.1/wavs" ${feat_dir}/melspec_new
    echo "Step 1: Done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Step 2: Split train, eval, test"
    python local/split_dataset.py ${download_dir}/LJSpeech-1.1/wavs ${feat_dir}
    echo "Step 2: Done"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Step 3: Train audio model"
    python ../../voicebox/train.py fit --config config/config.yaml
    echo "Step 3: Done"
fi
