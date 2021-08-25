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

| Dataset     | CIFAR 10        | CIFAR 10          | CIFAR 100       | CIFAR 100         | avg  |
| ----------- | --------------- | ----------------- | --------------- | ----------------- | ---- |
| **Model**   | ResNet (20->20) | ResNet (110->110) | ResNet (20->20) | ResNet (110->100) |      |
| baseline    | ResNet20        |                   |                 |                   |      |
| Logits      |                 |                   |                 |                   |      |
| ST          |                 |                   |                 |                   |      |
| AT          |                 |                   |                 |                   |      |
| ST + AT     |                 |                   |                 |                   |      |
| Tensorboard |                 |                   |                 |                   |      |



## How to apply to your dataset or model

pass



## Reference

- [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
- [Logits]()
- [ST]()
- [AT]()

