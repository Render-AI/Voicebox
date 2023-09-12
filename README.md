<div align="center">    
 
# Unofficial implementation of Voicebox

[![Paper](https://img.shields.io/badge/paper-arxiv.2306.15687-B31B1B.svg)](https://arxiv.org/abs/2306.15687)

</div>

## Description
Unofficial implementation of Voicebox.

Core codes from [lucidrains/voicebox-pytorch](https://github.com/lucidrains/voicebox-pytorch). I do not use voicebox-pytorch pypi release, instead put it in this repository just for convenient.

I did not trained duration model. It's on the TODO list.

## Demo

see [demo](./demo)


LJSpeech:

Original Text: Field agents supplement those on the detail, particularly when the <font color="#660000">President</font> is traveling.

Edited Text: Field agents supplement those on the detail, particularly when the <font color="#660000">Prime Minister</font> is traveling.

AIShell3:

Original Text: 夺得队史第五座 <font color="#660000">中</font>超冠军

Edited Text: 夺得队史第五座 <font color="#660000">英</font>超冠军


Note: aishell3's edited.wav is not good enough, because vocoder i used need more steps to converge.

## Checkpoint

see [LJSpeech](https://huggingface.co/omniking/Voicebox_LJSpeech)

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/chenht2010/Voicebox.git

# install dependeces
pip install lightning[extra] torch torchaudio tgt vocos torchdiffeq torchode einops beartype naturalspeech2-pytorch audiolm-pytorch
 ```   
 Next, navigate to examples, check README and run it.

## TODO
- [] try other universal vocoder
- [] try other alignment tools
- [] train duration model

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
