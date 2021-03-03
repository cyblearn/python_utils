# --------------------------------------------------------------   ----------------------------------------------------------------#
# 学习cnn
# 到底什么是一个层，层就是一种操作（padding，stride等）外加一堆参数。
# 比如torch.nn.Conv2d(in_channels, out_channels, kernel_size)
import torch

# step 1: 生成一个测试向量，也就是一张图
i = torch.ones(1,3,3)
# 生成batch的第四维, 这个nchw好像和rnn不同
i = torch.unsqueeze(i, 0)

# step 2: 生成卷积核
m = torch.nn.Conv2d(1,3,3,padding=0)
print(m.weight)
"""
m.weight.shape = torch.Size([3, 1, 3, 3])  # 注意这个为了方便计算，out通道是排在最前面的。决定了一个样本的输入通道是1，经过这个算子后输出通道为3。输出的wh由输入的wh和核共同决定。
tensor([[[[ 0.1070,  0.3077,  0.0875],
          [-0.2223, -0.2802, -0.2175],
          [-0.2160,  0.2391,  0.2871]]],


        [[[ 0.3214,  0.0017,  0.2324],
          [ 0.1403,  0.2240, -0.1238],
          [-0.2897, -0.2794,  0.1033]]],


        [[[-0.1449, -0.0218,  0.2492],
          [-0.1219, -0.1748,  0.1559],
          [-0.0441,  0.3233,  0.1930]]]], requires_grad=True)
"""
print(m.bias)
"""
torch.Size([3])
tensor([ 0.1321, -0.2557,  0.1825], requires_grad=True)
"""

# step3：前向传播
o=m(i)
print(o)
"""
o.shape = torch.Size([1, 3, 1, 1])  # 输出的n(batch)始终等于输入。c输出通道数是由卷积核Conv2D来决定的。
tensor([[[[0.2246]],

         [[0.0745]],

         [[0.5963]]]], grad_fn=<ThnnConv2DBackward>)
注意，o[0] = m.weight[0].sum() + m.bias[0]
"""
