## TextGrid

You can use textgrid from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), and you need read prepare.sh  and change it. 

or, you can do it by yourself, I suggest use conda env. This example will train MFA model from scratch, you need newer OS, CentOS 7 is failed to train.

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner=2.2.17 -y
conda actitvate aligner
bash prepare.sh
conda deactivate
```

## Vocoder
This example use Vocoder from [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)

I'll upload my trained weights.


## Mel mean and std

I already put mean and std in config/config.yaml, or you can just use the papaer's.
If you want calc mel mean and std:

```bash
python local/statistics.py
```

and change config.


## Train

```bash
bash run.sh
```

## Infer
```bash
bash infer.sh
```


pls see infer.py.


We have not train duration model, so if you want edited wav, you can edit TextGrid, change some phones, and make sure the changed phones's time section is equal or short than the mask section.



