# Knowledge Distillation
This repo contains implementation for basic Knowledge Distillation. I want to share training setting, summary and training log. I think you can reproduce same result by following tutorial. 



**Index**

1. Tutorial
2. Result Sharing
3. Test Accuracy Graph
4. Experiment Setting
5. References



## Tutorial

1. clone repo and install required library and download dataset

```bash
git clone
cd KnowledgeDistillation
pip3 install -r requirements.txt
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

4. you can choose training options using arguments showing below.

```bash
usage: main.py [-h] [-g GPU_ID] [-s SEED] [-m MODEL_NAME] [-d {cifar10,cifar100}] [-b BATCH_SIZE] [-w NUM_WORKERS] [-l LR] [-e NEPOCH]
               [-t TEACHER_MODEL] [-k {base,logit,st,at,at_st}] [-i ITER]

Knowledge Disillation

optional arguments:
  -h, --help            show this help message and exit
  -g GPU_ID, --gpu_id GPU_ID
                        Enter which gpu you want to use
  -s SEED, --seed SEED  Enter random seed
  -m MODEL_NAME, --model_name MODEL_NAME
                        Enter model name
  -d {cifar10,cifar100}, --dataset {cifar10,cifar100}
                        Enter dataset
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Enter batch size for train step
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        Enter the number of workers per dataloader
  -l LR, --lr LR        Enter learning rate
  -e NEPOCH, --nepoch NEPOCH
                        Enter the number of epoch
  -t TEACHER_MODEL, --teacher_model TEACHER_MODEL
                        Enter teacher model name
  -k {base,logit,st,at,at_st,fsp}, --kd_method {base,logit,st,at,at_st,fsp}
                        Enter knowledge Distillation Method
  -i ITER, --iter ITER  Enter the number of iteration

```



## Result Sharing

I cannot find any reported benchmark score yet. ([Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) could be option) So I just write down the test results. 

- If you click the summary next to method name, it will link to the page for summarizing methods with training results. 
- If you click the tensorboard next to method name, you can see the (train/test) (accuracy/loss) in tensorboard.
- scores are calculated by averaging 3 times results
- This scores is just my scores done in this work, not official one.
- All model are trained with same settings, which are shown in [Experiment Setting](#experiment-setting) section
- **[FSP]** I had trained FSP with original paper settings(2 stage train), but the result was not good. So I train model with setting as same as others.

| Dataset                                | CIFAR 10       | CIFAR 10       | CIFAR 100      | CIFAR 100      | avg            |
| -------------------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| **Model(ResNet)**                      | 20 >> 20       | 110 >> 110     | 20 >> 20       | 110 >> 110    |                |
| baseline <br />([tensorboard](https://tensorboard.dev/experiment/uDRWaW9bQ7qD4RTVoMijpQ/#scalars), [summary](docs/baseline.md)) | 92.0           | 92.5           | 67.6           | 71.6          | 80.9           |
| Logits <br />([tensorboard](https://tensorboard.dev/experiment/XQxR2I61QoergHwPQK2jxg), [summary](docs/logit.md)) | 92.5(+0.5)     | **94.2(+1.7)** | 69.2(+1.6)     | 71.8(+0.2)     | 81.9(+1.0)     |
| **ST** <br />([tensorboard](https://tensorboard.dev/experiment/wBPnPMRtQ6mjjio2oZckQA/), [summary](docs/st.md)) | **92.8(+0.8)** | 93.9(+1.4)     | **69.9(+2.3)** | **74.1(+2.5)** | **82.7(+1.8)** |
| AT <br />([tensorboard](https://tensorboard.dev/experiment/wBPnPMRtQ6mjjio2oZckQA/), [summary](docs/at.md)) | 91.9(-0.1)     | 93.3(+0.8)     | 68.4(+0.8)     | 72.6(+1.0)     | 81.55(+0.65)   |
| ST + AT <br />([tensorboard](https://tensorboard.dev/experiment/TWk1w7R5RZ6SmD3n6tVd3w/), [summary](docs/at_st.md)) | 92.6(+0.6)     | *93.7(+1.2)     | 68.7(+1.1)     | *73.7(+2.1)     | 82.2(+1.5)     |
| FSP<br />([tensorboard](https://tensorboard.dev/experiment/zMKtJqwKRJGfXUEAFMXqhw/), [summary](docs/fsp.md)) | 92.0(+0.0) | 92.7(+0.2) | 68.0(+0.4) | 72.8(+1.2) | 81.4(+0.5) |

*\* means 2 times average scores*



## Test Accuracy Graph

Cifar100/110->110 

![image](https://user-images.githubusercontent.com/31476895/131270339-9fcea168-8a73-44bc-bddf-c6ec59983106.png)



## Experiment Setting

1. Dataset Description

| Name       | # Train Image | # Test Image |
| ---------- | ------------- | ------------ |
| Cifar10    | 50000         | 10000        |
| Cifar100   | 50000         | 10000        |
| Resoultion | 32x32         | 32x32        |



2. Model Description

|                   | ResNet 20                                                    | ResNet 110                                                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| # channel         | [16, 32, 64]                                                 | [16, 32, 64]                                                 |
| # block           | [3, 3, 3]                                                    | [18, 18, 18]                                                 |
| feature map shape | [(32x32), (16x16), (8x8)]                                    | [(32x32), (16x16), (8x8)]                                    |
| ect               | No Stride in conv1<br />No Max Pooling at input layer<br /># Layer is 3 | No Stride in conv1<br />No Max Pooling at input layer<br /># Layer is 3 |



3. Experiment Details

| Range         | Category           | Content                                                      |
| ------------- | ------------------ | ------------------------------------------------------------ |
| Data          | Data Preprocessing | Normalize                                                    |
|               | Train Augmentation | Pad(4) & Random Crop, Random Horizontal Flip                 |
|               | Test Augmentation  | Center Crop                                                  |
| Model         | Model Structure    | ResNet 20, 110 (ResNet 32x32 version, proposed in original paper) |
|               | Regularization     | Init parameter<br />`kaiming_normal_` for conv<br />`xavier_uniform_` for linear<br />Batch Normalization |
| Training Tool | Optimizer          | SGD, lr=0.1, weight_decay=1e-4, momentum = 0.9               |
|               | Criterion          | Basic is cross_entropy loss, additional loss for each method |
|               | LR Scheduler       | Multi-Step LR, milestones=[100, 150]                         |
| Train         | epoch              | 200                                                          |
|               | Batch size         | 128                                                          |
| Evaluation    | evaluation method  | Best verification result                                     |



## Reference

- [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
- [szagoruko/attention-transfer](https://github.com/szagoruyko/attention-transfer)
- [2014, Do Deep Nets Really Need to be Deep(Logits)](https://arxiv.org/abs/1312.6184)
- [2015, Distilling the knowledge in a Neural Network(ST)](https://arxiv.org/abs/1503.02531)
- [2016, Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer(AT)](https://arxiv.org/abs/1612.03928)
- [2017, A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)

