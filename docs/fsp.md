---
typora-copy-images-to: pics
---

## FSP Summary

This method is really similar with Attention Transfer (AT). The only difference is how it process activation map before compare each other. The way of AT is just calculate mean of all activation map. But, in FSP, you should calculate fsp matrix first, which is inner product of two different dimensional matrix. If you are interested in how to compute fsp matrix, you can see code in below picture or go to original paper.

![image-20210901082916129](pics/image-20210901082916129.png)



### Reference

- [2017, A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)