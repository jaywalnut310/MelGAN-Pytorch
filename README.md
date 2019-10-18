# ClariNet
A Pytorch Implementation of MelGAN (Mel Spectrogram --> Waveform)


# Requirements

PyTorch 1.2.0 & python 3.6 & Librosa

# Examples

#### Step 1. Download Dataset

- LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech`

#### Step 3. Train MelGAN

-c: configurations & hyper parameteres in json
-m: model directory name

`python train.py -c configs/base.json -m test

#### Step 4. Synthesize

Run [Sample Test.ipynb](./Sample%20Test.ipynb)

While improving, I share a temporary checkpoint of generator, which only runs about 140K steps: [link](https://drive.google.com/open?id=1vBKtGwR4n0rw0VqfuybC5Obd8BqJEGcL)

# References

- MelGAN: [https://arxiv.org/abs/1910.06711](https://arxiv.org/abs/1910.06711)
- Base codes are higly adopted from: [https://github.com/ksw0306/ClariNet](https://github.com/ksw0306/ClariNet)
