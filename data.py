"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import CustomDataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

# 加载数据集
train_raw = Multi30k(split='train', language_pair=('de', 'en'))
valid_raw = Multi30k(split='valid', language_pair=('de', 'en'))

# 设置分词器
tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')
tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')

# 初始化自定义数据集
train_dataset = CustomDataset(
    train_raw, 
    src_tokenize=tokenize_de, 
    tgt_tokenize=tokenize_en,
    min_freq=2, 
    src_pad_size=max_len, 
    target_pad_size=max_len
)
valid_dataset = CustomDataset(
    valid_raw,
    src_tokenize=tokenize_de,
    tgt_tokenize=tokenize_en,
    min_freq=2,
    src_pad_size=max_len,
    target_pad_size=max_len
)

# 创建 DataLoader，直接传入自定义数据集
batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# 构建词汇表（只需用训练集）
src_vocab = train_dataset.src_vocab
tgt_vocab = train_dataset.tgt_vocab

# 获取特殊符号对应的索引
src_pad_idx = src_vocab[train_dataset.pad_token]
tgt_pad_idx = tgt_vocab[train_dataset.pad_token]
tgt_sos_idx = tgt_vocab[train_dataset.sos_token]
