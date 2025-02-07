"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import CustomDataLoader
from util.tokenizer import Tokenizer
from torchtext.datasets import Multi30k

tokenizer = Tokenizer()
loader = CustomDataLoader(ext=('.en', '.de'),
                    init_token='<sos>',
                    eos_token='<eos>')

# 加载数据集：注意，Multi30k返回的是迭代器，需重新实例化
train_iter = Multi30k(split='train', language_pair=('en', 'de'))
valid_iter = Multi30k(split='valid', language_pair=('en', 'de'))
test_iter = Multi30k(split='test', language_pair=('en', 'de'))

# 构建词汇表（只需用训练集）
src_vocab, tgt_vocab = loader.build_vocab(train_iter)

# 获取转换函数
src_transform, tgt_transform = loader.get_transform()

# 构建索引化后的数据集
train_dataset = [(src_transform(en), tgt_transform(de)) for en, de in Multi30k(split='train', language_pair=('en', 'de'))]
valid_dataset = [(src_transform(en), tgt_transform(de)) for en, de in Multi30k(split='valid', language_pair=('en', 'de'))]

# 创建数据加载器
train_dataloader = loader.make_iter(train_dataset, batch_size=batch_size, device=device)
valid_dataloader = loader.make_iter(valid_dataset, batch_size=batch_size, device=device)

# 获取特殊符号对应的索引
src_pad_idx = loader.src_vocab[loader.pad_token]
trg_pad_idx = loader.tgt_vocab[loader.pad_token]
trg_sos_idx = loader.tgt_vocab[loader.init_token]

enc_voc_size = len(loader.src_vocab)
dec_voc_size = len(loader.tgt_vocab)
