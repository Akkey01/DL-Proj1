# MeanSquareTerror_P1
Deep Learning Project 1

A private fork of [pytorch-cifar](url).

Train with
```
python3 train.py --exp_name exp1
```

Get CIFAR-10 test accuracy with
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name>
```

Generate submission CSV for the no-label data with 
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name> --nolabel 1
```

Carry out test-time augmentation with 
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name> --tta 1
```