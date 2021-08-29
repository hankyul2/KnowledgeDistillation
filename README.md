# Knowledge Distillation (WIP)
This repo contains basic implementation for Knowledge Distillation. I want you to get help from this work.

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
  -k {base,logit,st,at,at_st}, --kd_method {base,logit,st,at,at_st}
                        Enter knowledge Distillation Method
  -i ITER, --iter ITER  Enter the number of iteration

```



## Result Sharing

I cannot find any reported benchmark score yet. ([Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) could be option) So I just write down the test results. 

- If you click the summary next to method name, it will link to the page for summarizing methods with training results. 
- If you click the TensorBoard, you can see the training and test log for each benchmark in online.
- scores are calculated by averaging 3 times results
- This scores is just my scores done in this work, not official one.

| Dataset                                | CIFAR 10       | CIFAR 10       | CIFAR 100      | CIFAR 100      | avg            |
| -------------------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| **Model(ResNet)**                      | 20 >> 20       | 110 >> 110     | 20 >> 20       | 110 >> 100     |                |
| baseline ([summary](docs/baseline.md)) | 92.0           | 92.5           | 67.6           | 71.6          | 80.9           |
| Logits ([summary](docs/logit.md))      | 92.5(+0.5)     | **94.2(+1.7)** | 69.2(+1.6)     | 71.8(+0.2)     | 81.9(+1.0)     |
| **ST** ([summary](docs/st.md))         | **92.8(+0.8)** | 93.9(+1.4)     | **69.9(+2.3)** | **74.1(+2.5)** | **82.7(+1.8)** |
| AT ([summary](docs/at.md))             | 91.9(-0.1)     | 93.3(+0.8)     | 68.4(+0.8)     | 72.6(+1.0)     | 81.55(+0.65)   |
| ST + AT ([summary](docs/at_st.md))     | 92.6(+0.6)     | *93.7(+1.2)     | 68.7(+1.1)     | *73.7(+2.1)     | 82.2(+1.5)     |
| TensorBoard                            |                |                |                |                |                |

*\* means not 2 times average scores*

## How to apply to your dataset or model

pass



## Reference

- [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
- [szagoruko/attention-transfer](https://github.com/szagoruyko/attention-transfer)
- [2014, Do Deep Nets Really Need to be Deep(Logits)](https://arxiv.org/abs/1312.6184)
- [2015, Distilling the knowledge in a Neural Network(ST)](https://arxiv.org/abs/1503.02531)
- [2016, Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer(AT)](https://arxiv.org/abs/1612.03928)

