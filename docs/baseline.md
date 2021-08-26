### Baseline Summary

ResNet20, 110 architecture are used as baseline model. We use simple and light version of ResNet. Channel  number, the number of blocks are shown in below table.(which is suggested at [original paper](https://arxiv.org/abs/1512.03385) and I reference [Knowledge Distillation Zoo repo](https://github.com/AberHu/Knowledge-Distillation-Zoo) implementation)

 

|                   | ResNet 20                                                    | ResNet 110                                                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| # channel         | [16, 32, 64]                                                 | [16, 32, 64]                                                 |
| # block           | [3, 3, 3]                                                    | [18, 18, 18]                                                 |
| feature map shape | [(32x32), (16x16), (8x8)]                                    | [(32x32), (16x16), (8x8)]                                    |
| ect               | No Stride in conv1<br />No Max Pooling at input layer<br /># Layer is 3 | No Stride in conv1<br />No Max Pooling at input layer<br /># Layer is 3 |



### Training Log ?

I do not know this can be helpful for you, but I share this anyway. Scores in main ReadMe is calculated from this table.

I have full training log and tensorboard log also. I will share them in later.

1. ResNet20/Cifar10

| no   | method | dataset | model        | start_time          | acc     | epoch | nepoch | lr   | batch_size |
| ---- | ------ | ------- | ------------ | ------------------- | ------- | ----- | ------ | ---- | ---------- |
| 1    | base   | cifar10 | resnet_32_20 | 2021-08-26/17-52-28 | 92.0573 | 165   | 200    | 0.1  | 128        |
| 2    | base   | cifar10 | resnet_32_20 | 2021-08-26/18-30-31 | 91.9972 | 199   | 200    | 0.1  | 128        |
| 3    | base   | cifar10 | resnet_32_20 | 2021-08-26/19-08-41 | 91.8369 | 178   | 200    | 0.1  | 128        |

2. ResNet20/Cifar100

| no   | method | dataset  | model        | start_time          | acc     | epoch | nepoch | lr   | batch_size |
| ---- | ------ | -------- | ------------ | ------------------- | ------- | ----- | ------ | ---- | ---------- |
| 1    | base   | cifar100 | resnet_32_20 | 2021-08-26/17-52-29 | 67.528  | 165   | 200    | 0.1  | 128        |
| 2    | base   | cifar100 | resnet_32_20 | 2021-08-26/18-29-54 | 67.7284 | 158   | 200    | 0.1  | 128        |
| 3    | base   | cifar100 | resnet_32_20 | 2021-08-26/19-07-08 | 67.5982 | 196   | 200    | 0.1  | 128        |

3. ResNet110/Cifar10

| no   | method | dataset | model         | start_time          | acc     | epoch | nepoch | lr   | batch_size |
| ---- | ------ | ------- | ------------- | ------------------- | ------- | ----- | ------ | ---- | ---------- |
| 1    | base   | cifar10 | resnet_32_110 | 2021-08-26/17-52-29 | 91.8269 | 188   | 200    | 0.1  | 128        |
| 2    | base   | cifar10 | resnet_32_110 | 2021-08-26/19-42-46 | 93.0689 | 196   | 200    | 0.1  | 128        |
| 3    | base   | cifar10 | resnet_32_110 | 2021-08-26/21-25-25 | 92.6482 | 186   | 200    | 0.1  | 128        |

4. ResNet110/Cifar100

| no   | method | dataset  | model         | start_time          | acc     | epoch | nepoch | lr   | batch_size |
| ---- | ------ | -------- | ------------- | ------------------- | ------- | ----- | ------ | ---- | ---------- |
| 1    | base   | cifar100 | resnet_32_110 | 2021-08-26/17-52-29 | 71.264  | 105   | 200    | 0.1  | 128        |
| 2    | base   | cifar100 | resnet_32_110 | 2021-08-26/19-40-24 | 71.7348 | 195   | 200    | 0.1  | 128        |
| 3    | base   | cifar100 | resnet_32_110 | 2021-08-26/21-22-09 | 71.7648 | 106   | 200    | 0.1  | 128        |



#### References

- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) 
- [Knowledge Distillation Zoo repo](https://github.com/AberHu/Knowledge-Distillation-Zoo)