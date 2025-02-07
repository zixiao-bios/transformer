"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# device = torch.device('cpu')
# device = torch.device('cuda')
device = torch.device('mps')

# data parameter setting
batch_size = 128

# 序列最大长度
max_len = 50

d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
