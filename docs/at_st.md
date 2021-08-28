### AT_ST Summary

AT + ST is just combine loss of at and st. All method are same, just one things is different.(alpha=0.9)

### AT_ST Train Result

ResNet20>>20/Cifar10 (92.6)

| 1    | method | dataset | student      | teacher      | start_time          | acc         | epoch | nepoch | lr   | batch_size |
| ---- | ------ | ------- | ------------ | ------------ | ------------------- | ----------- | ----- | ------ | ---- | ---------- |
| 29   | at_st  | cifar10 | resnet_32_20 | resnet_32_20 | 2021-08-27/08-26-17 | 92.53806305 | 153   | 200    | 0.1  | 128        |
| 38   | at_st  | cifar10 | resnet_32_20 | resnet_32_20 | 2021-08-27/10-29-39 | 92.54808044 | 179   | 200    | 0.1  | 128        |
| 45   | at_st  | cifar10 | resnet_32_20 | resnet_32_20 | 2021-08-27/12-33-10 | 92.74839783 | 157   | 200    | 0.1  | 128        |

ResNet20>>20/Cifar100 (68.7)

| 1    | method | dataset  | student      | teacher      | start_time          | acc         | epoch | nepoch | lr   | batch_size |
| ---- | ------ | -------- | ------------ | ------------ | ------------------- | ----------- | ----- | ------ | ---- | ---------- |
| 28   | at_st  | cifar100 | resnet_32_20 | resnet_32_20 | 2021-08-27/08-26-19 | 68.61978912 | 169   | 200    | 0.1  | 128        |
| 37   | at_st  | cifar100 | resnet_32_20 | resnet_32_20 | 2021-08-27/10-29-04 | 68.55970001 | 192   | 200    | 0.1  | 128        |
| 44   | at_st  | cifar100 | resnet_32_20 | resnet_32_20 | 2021-08-27/12-32-35 | 68.84014893 | 188   | 200    | 0.1  | 128        |

ResNet110>>110/Cifar10 (93.7)

| no   | method | dataset | student       | teacher       | start_time          | acc      | epoch | nepoch | lr   | batch_size |
| ---- | ------ | ------- | ------------- | ------------- | ------------------- | -------- | ----- | ------ | ---- | ---------- |
| 1    | at_st  | cifar10 | resnet_32_110 | resnet_32_110 | 2021-08-27/08-26-22 | 93.44952 | 173   | 200    | 0.1  | 128        |
| 2    | at_st  | cifar10 | resnet_32_110 | resnet_32_110 | 2021-08-27/14-49-38 | 93.91026 | 173   | 200    | 0.1  | 128        |

ResNet110>>110/Cifar100 (73.7)

| no   | method | dataset  | student       | teacher       | start_time          | acc      | epoch | nepoch | lr   | batch_size |
| ---- | ------ | -------- | ------------- | ------------- | ------------------- | -------- | ----- | ------ | ---- | ---------- |
| 1    | at_st  | cifar100 | resnet_32_110 | resnet_32_110 | 2021-08-27/08-26-22 | 73.59776 | 193   | 200    | 0.1  | 128        |
| 2    | at_st  | cifar100 | resnet_32_110 | resnet_32_110 | 2021-08-27/14-49-35 | 74.13863 | 157   | 200    | 0.1  | 128        |
| 3    | at_st  | cifar100 | resnet_32_110 | resnet_32_110 | 2021-08-27/18-24-54 | 73.31731 | 191   | 200    | 0.1  | 128        |