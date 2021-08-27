# Knowledge Distillation (WIP)
This repo contains basic implementation for Knowledge Distillation. I want you to get help from this work.

## Tutorial

1. clone repo and download dataset

```bash
git clone
python3 download_dataset.py
```

2. Train teacher model (ResNet 20)

```bash
python3 main.py -g 0 -m resnet_32_20 -d cifar10
```

3. Train student model with some help from teacher model

```bash
python3 main.py -g 0 -m resnet_32_20 -d cifar10 -t resnet_32_20 -k at
```



## Result Sharing

I cannot find any reported benchmark score yet. ([Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) could be option) So I just write down the test results. 

- If you click the method name, it will link to the page for summarizing methods with summary of original paper and applied result. 
- If you click the TensorBoard, you can see the training and test log for each benchmark in online.
- scores are calculated by averaging 3 times results
- This scores is just my scores done in this work, not official ones.

| Dataset           | CIFAR 10       | CIFAR 10        | CIFAR 100      | CIFAR 100       | avg          |
| ----------------- | -------------- | --------------- | -------------- | --------------- | ------------ |
| **Model(ResNet)** | 20 >> 20       | 110 >> 110      | 20 >> 20       | 110 >> 100      |              |
| baseline          | 92.0           | 92.5            | 67.6           | *71.6           | 80.9         |
| Logits            | 92.5(+0.5)     | ***94.2(+1.7)** | 69.2(+1.6)     |                 |              |
| **ST**            | **92.8(+0.8)** | *93.5(+1.0)     | **69.9(+2.3)** | ***74.2(+2.6)** | 82.6(+1.7)   |
| AT                | 91.9(-0.1)     | *93.2(+0.7)     | 68.4(+0.8)     | *72.7(+1.1)     | 81.55(+0.65) |
| ST + AT           | 92.6(+0.6)     |                 | 68.7(+1.1)     |                 |              |
| Tensorboard       |                |                 |                |                 |              |

*\* means not 3 times average scores*

## How to apply to your dataset or model

pass



## Reference

- [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
- [Logits]()
- [ST]()
- [AT]()

