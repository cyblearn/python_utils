# 学习cnn
# 到底什么是一个层，层就是一种操作（padding，stride等）外加一堆参数。
# 比如torch.nn.Conv2d(in_channels, out_channels, kernel_size)
import torch

# -------------------------------------------------------------- torch.nn.Conv2d 算子  ----------------------------------------------------------------#

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

# -------------------------------------------------------------- torch.nn.ReLU 算子  ----------------------------------------------------------------#
"""
逐元素处理，输入是多少维度，输出也是多少维度。
"""
i = torch.randn(5,5)          # 正态分布
m=torch.nn.ReLU()
o=m(i)
"""
print(i)
tensor([[-2.5638, -0.7381, -1.1224, -1.5505,  0.5580],
        [ 2.1287,  0.2674, -0.0580,  1.5029,  1.1543],
        [ 1.3154,  0.3611, -0.9056, -0.9017,  0.0924],
        [-0.6977,  0.6132,  0.8106,  0.6332, -1.9930],
        [ 0.1393, -0.2282,  0.1238,  0.7585, -0.5896]])
print(o)
tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.5580],
        [2.1287, 0.2674, 0.0000, 1.5029, 1.1543],
        [1.3154, 0.3611, 0.0000, 0.0000, 0.0924],
        [0.0000, 0.6132, 0.8106, 0.6332, 0.0000],
        [0.1393, 0.0000, 0.1238, 0.7585, 0.0000]])

"""

# -------------------------------------------------------------- torch.nn.Linear 算子  ----------------------------------------------------------------#
"""
不管输入有多少个维度，Linear总是把最后一个维度做线性变换。
比如输入size为[2,1,4]，Linear的shape是[4,3]，则输出的shape是[2,1,3]。
还要注意，y=Ax+b, 所以Linear的weight里存的实际是A
一般在输入前需要通过，var = var.view(var.size(0), -1)，来按第一个样本维度来把剩下的维度展平。-1表示自动推导后续维度
"""
i=torch.ones(1,4)
m=torch.nn.Linear(4, 3)
o=m(i)
"""
m.weight.shape = torch.Size([3, 4])
m.weight = 
tensor([[-0.0917, -0.1970,  0.1640, -0.3665],                                   # !!!! 注意看数据的内存排列顺序，就是按这个表顺序排列，一行一行，一个维度一个维度排
        [-0.1683,  0.1646, -0.2970, -0.1425],
        [-0.2873,  0.2123, -0.4675,  0.3845]], requires_grad=True)

m.bias.shape = torch.Size([3])
tensor([-0.1071,  0.2867, -0.3275], requires_grad=True)

o.shape = torch.Size([1, 3])
o = tensor([[-0.5983, -0.1565, -0.4855]])

"""

# -------------------------------------------------------------- torch.nn.MaxPool2d 算子  ----------------------------------------------------------------#
"""
2d的最大池化算子。就是操作一个张量的最后两个维度。输入的前几个维度保持
"""
i = torch.randn(2,4,4)
m = torch.nn.MaxPool2d(kernel_size=2, stride=2)
o = m(i)
"""
i = 
tensor([[[ 0.3665,  1.7927, -0.4443,  1.1879],
         [-0.1447,  0.7678,  0.7669, -0.2975],
         [ 1.5361,  1.1066,  1.9095, -0.4918],
         [-0.3698, -0.9769,  0.5337,  0.9988]],

        [[ 0.6057,  0.3738, -0.0050, -1.2049],
         [-0.7801,  0.6768,  0.2231,  0.1022],
         [ 0.6095,  0.7005, -0.3676,  1.2724],
         [ 2.4264,  0.0223,  0.2415,  0.9974]]])
o.shape = torch.Size([2, 2, 2])
o = 
tensor([[[1.7927, 1.1879],
         [1.5361, 1.9095]],

        [[0.6768, 0.2231],
         [2.4264, 1.2724]]])
"""

# -------------------------------------------------------------- torch.nn.Dropout与Dropout2d 算子  ----------------------------------------------------------------#
"""
torch.nn.Dropout(p=0.5): 逐元素按概率丢弃，也就是从左到右最后一个维度。
torch.nn.Dropout2d(p=0.5)：按通道进行丢弃，也就是从左到右的第二个通道。
看这个例子：https://blog.csdn.net/appleml/article/details/88670580
"""
m = nn.Dropout(p=0.5)
n = nn.Dropout2d(p=0.5)
input = torch.randn(1, 2, 6, 3)

print(m(input))
print('****************************************************')
print(n(input))

# -------------------------------------------------------------- torch.nn.Dropout与Dropout2d 算子  ----------------------------------------------------------------#
"""
softmax一般针对二维数据，每行表示一个样本，每列表示不同的维度。dim=0表示按列计算；dim=1表示按行计算。
返回结果是一个与输入shape相同的张量，每个元素的取值范围在（0,1）区间。
比如：
输入：大小batch_size为4，类别数为6的向量
batch_size = 4
class_num = 6
inputs = torch.randn(batch_size, class_num)
看这篇博客：https://zhuanlan.zhihu.com/p/137791367
"""
batch_size = 4
class_num = 6
inputs = torch.randn(batch_size, class_num)
for i in range(batch_size):
    for j in range(class_num):
        inputs[i][j] = (i + 1) * (j + 1)

Softmax = torch.nn.Softmax(dim=1)
probs = Softmax(inputs)
print("probs:\n", probs)



# --------------------------------------------------------------  循环网络部分  --------------------------------------------------------------- #
# 看这篇文章：https://zhuanlan.zhihu.com/p/71732459
# 这里要啰嗦一句，karpathy在RNN的前向中还计算了一个输出向量output vector，
# 但根据RNN的原始公式，它的输出只有一个hidden_state，至于整个网络最后的output vector，
# 是在hidden_state之后再接一个全连接层得到的，所以并不属于RNN的内容。
# 包括pytorch和tf框架中，RNN的输出也只有hidden_state。理解这一点很重要。
class RNN:
  def __init__(self, hidden_size = 5, input_size=6):
    self.W_hh = np.ones((hidden_size, hidden_size))
    self.W_xh = np.ones((hidden_size, input_size))
  
  
  def step(self, x, hidden):
    # update the hidden state
    # 只要给定了input_size和hidden的size，则W_hh及W_xh的size均已知
    hidden = np.tanh(np.dot(self.W_hh, hidden) + np.dot(self.W_xh, x))
    return hidden

input_size = 6
hidden_size = 5

# x: [seq_len * input_size]
x = np.ones((12, input_size))
seq_len = x.shape[0]


rnn = RNN(hidden_size, input_size)
    

# 初始化一个hidden_state，RNN中的参数没有包括hidden_state，
# 只包括hidden_state对应的权重W和b，
# 所以一般我们会手动初始化一个全零的hidden_state
hidden_state = np.zeros((hidden_size))

# 下面这个循环就是RNN的工作流程了，看到没有，每次输入的都是一个时间步长的数据，
# 然后同一个hidden_state会在循环中反复输入到网络中。
print(seq_len)
for i in range(seq_len):
    hidden_state = rnn.step(x[i, :], hidden_state)
    print(i)
    print(hidden_state)

# --------------------------------------------------------------  RNN的层数和输出  -----------------------------------------------------------  #
# https://blog.csdn.net/qq_41295081/article/details/113752719
# 输出：
# ht：最后一个时间戳上面所有的 memory 状态
# out：所有时间戳上的最后一个 memory 状态
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
x = torch.randn(10, 3, 100)
out, h_t = rnn(x)
print(out.shape) # [10, 3, 20]
print(h_t.shape) # [4, 3, 20]
# 解释下例子的输入输出：
# 其中RNN的参数为input_size即单个样本的尺寸大小为100,hidden_size即隐藏层中输出特征的大小为20，num_layers即纵向的隐藏层个数为4
# input (seq_len, batch, input_size)---> (10，3，100)
# output  (seq_len, batch, num_directions * hidden_size)--> (10，3，20)
# h_t     (num_layers * num_directions, batch, hidden_siz) --> (4，3，20)
