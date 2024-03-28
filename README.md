# Reg_Seg readme

这是一个基于2024_MIA_Ana这篇论文的一个代码，相比较于之前的，主要是引入了seg信息，另外这和原文的描述并不一样，原文用的方法是基于Transformer的，这里直接用的卷积。下面介绍一下各个文件

### train.py
train.py 是整个程序的入口，里面设计了一些运行一个深度学习程序该有的设置。

### train_seg.py

这是为了验证分割部分正确与否，单独把分割部分拿出来训练的train文件

### train_nocross.py

这也相当于一个消融实验，去掉交叉注意力的部分

### dataset.py
文件的读取，dataset。返回的内容如下：

```python
return {
    'fixed': fixed,
    'moving': moving,
    'fixed_mask' : fixed_mask,
    'moving_mask':moving_mask
}
```
### without_cross.py
里面有个AC_DMiR_without_cross_attention类，是去掉交叉注意力的模型


# 新学期修改
优化了初始配准部分
1. 首先是`network_reg.py`.这是不带交叉注意力的初始配准，和voxelmorph的原理一致，运行的模型以及中间结果保存在 `/media/user_gou/Elements/Shi/Reg_Seg/result_without_cross_attention`文件夹


2. 在`network_reg.py`的基础上加入交叉注意力，得到`network_reg_with_cross_attention.py`。运行的模型以及中间结果保存在 `/media/user_gou/Elements/Shi/Reg_Seg/result_with_cross_attention`文件夹

## 实验记录 3/21组会
### results1
`python train.py`
`model = AC_DMiR()`
前面的配准网络和分割网络并没有带cross attention，而是在bottleneck拿出来后才进行的cross attention，然后输入到后面的网络。
