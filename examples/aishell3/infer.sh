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


MFA_dir=data/MFA

test_dir=test



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Step 0: Get aligned transcript and phones"
    mfa align ${test_dir}/wavs ${MFA_dir}/lexicon.txt ${MFA_dir}/chinese.zip ${test_dir}/TextGrid
    python local/textgrid2lab.py "${test_dir}/TextGrid" ${test_dir}
    echo "Step 0: Done"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Step 1: Get melspec"
    echo "Step 1: Done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Step 2: Split train, eval, test"
    python local/gen_test_dataset.py --unaligned ${test_dir}/wavs ${test_dir}
    echo "Step 2: Done"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Step 3: Generate audio"
    python infer.py --test_path ${test_dir}
    echo "Step 3: Done"
fi
