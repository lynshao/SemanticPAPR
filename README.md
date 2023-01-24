# Semantic Communications with Discrete-time Analog Transmission: A PAPR Perspective

This repository is the official implementation of paper [Semantic Communications with Discrete-time Analog Transmission: A PAPR Perspective] (https://arxiv.org/abs/2208.08342).

## Requirements

Experiments were conducted on Python 3.8.5. To install requirements:

```setup
pip install -r requirements.txt
```

## Running
```train
python main.py --BatchSize {} --snr {} --precoding {} --mapping {} --lamb {} --thres {} --clip {} --fading {} --hstd {}
```

Example:

OFDM, no PAPR loss, no clipping, AWGN channel

python main.py --snr 10 --numepoch 100 --precoding 0 --mapping 0 --lamb 0 --thres 0 --clip 0 --fading 0




> If you find this repository useful, please kindly cite as
> 
> @article{ShaoSemanticPAPR,
> 
> title={Semantic Communications with Discrete-time Analog Transmission: A PAPR Perspective},
> 
> author={Shao, Yulin and Gunduz, Deniz},
> 
> journal={IEEE Wireless Communications Letters},
> 
> year={2022}
> 
> }
