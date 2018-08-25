import torch
import torch.nn as nn
from mentality import conv_output_shape, conv_transpose_output_shape


x= torch.ones(10,1,400,600)
c2d = nn.Conv2d(1, 1, kernel_size=5, stride=2)
y = c2d(x)
print(y.shape)
print(conv_output_shape((x.shape[2],x.shape[3]), kernel_size=5, stride=2))