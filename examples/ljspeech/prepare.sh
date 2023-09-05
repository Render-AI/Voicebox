#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

. ./path.sh || exit 1;


# General configuration
stage=0              # Processes starts from the specified stage.
stop_stage=5     # Processes is stopped at the specified stage.

download_dir=downloads

feat_dir=data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Step 0: Download MFA model"
    mkdir -p ${download_dir}/MFA
    wget  https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -O ${download_dir}/MFA/english.zip
    echo "Step 0: Done"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Step 1a: Download dataset and extract"
    mkdir -p ${download_dir}
    wget --no-check-certificate https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -O "${download_dir}/LJSpeech-1.1.tar.bz2"
    echo "Extracting ..."
    tar -xf "${download_dir}/LJSpeech-1.1.tar.bz2" -C "${download_dir}"

    echo "Step 1b: generate transcript for MFA"
    python local/prepare_transcript.py ${download_dir}/LJSpeech-1.1/metadata.csv ${download_dir}/LJSpeech-1.1/wavs
    echo "Step 1: Done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Step 2: Download lexicon and process"
    wget --no-check-certificate  http://www.openslr.org/resources/11/librispeech-lexicon.txt -O ${download_dir}/MFA/librispeech-lexicon.txt
    python local/prepare_lexicon.py ${download_dir}/MFA/librispeech-lexicon.txt ${download_dir}/MFA/modified_librispeech-lexicon.txt
    echo "Step 2: Done"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Step 3: MFA, if error, pls check ./temp dir"
    mfa align -t ./temp -j 4  ${download_dir}/LJSpeech-1.1/wavs ${download_dir}/MFA/modified_librispeech-lexicon.txt ${download_dir}/MFA/english.zip ${feat_dir}/TextGrid
    rm -rf ./temp
    echo "Step 3: Done"
fi

