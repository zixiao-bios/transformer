# Python
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, raw_data, src_tokenize, tgt_tokenize, src_pad_size, target_pad_size, min_freq=2):
        """
        Args:
            raw_data: 文本对列表，每个元素为 (src_text, tgt_text)
            src_tokenize: 源语言分词器
            tgt_tokenize: 目标语言分词器
            min_freq: 最小词频
            src_pad_size: 指定源序列填充的固定长度
            target_pad_size: 指定目标序列填充的固定长度
        """
        self.raw_data = list(raw_data)
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = tgt_tokenize
        self.min_freq = min_freq
        self.src_pad_size = src_pad_size
        self.tgt_pad_size = target_pad_size
        
        # 特殊 token 定义
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        # 构建词汇表
        self.src_vocab, self.tgt_vocab = self.build_vocab(self.raw_data)
        
        print(f'完成数据集初始化，样本数量: {len(self.raw_data)}, 源语言词汇表大小: {len(self.src_vocab)}, 目标语言词汇表大小: {len(self.tgt_vocab)}')
    
    def _tokenize(self, data_iter, tokenizer, idx, add_special=True):
        """分词函数，返回分词后的结果，可选择是否添加特殊 token"""
        for data in data_iter:
            tokens = tokenizer(data[idx])
            if add_special:
                yield [self.sos_token] + tokens + [self.eos_token]
            else:
                yield tokens
    
    def build_vocab(self, data):
        """
        利用 build_vocab_from_iterator 构建源和目标语言词汇表
        Args:
            data: 原始数据列表，每个元素为 (src_text, tgt_text)
        Returns:
            src_vocab, tgt_vocab
        """
        # 构建源语言词汇表
        # 使用 torchtext 的 build_vocab_from_iterator 函数构建源语言的词汇表
        # 参数解释：
        #   - 第一个参数是一个生成器，通过 _tokenize 方法遍历训练集中的文本数据并进行分词，其中 idx=0 表示选择数据中的第一个元素（源语言句子）。
        #   - min_freq 用于设置词汇的最小词频，词频低于该值的词将不会进入词汇表
        #   - specials 列表定义了一些特殊的 token，例如填充符 (<pad>)、未知词 (<unk>)、句子开始 (<sos>) 和句子结束 (<eos>)
        src_vocab = build_vocab_from_iterator(
            self._tokenize(data, self.src_tokenize, idx=0),
            min_freq=self.min_freq,
            specials=[self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        )
        
        # 设置默认索引：当词汇表中不存在某个 token 时，返回默认索引
        # 这样可以保证在文本转换为数值序列时，如果遇到未收录的词，会返回一个预设的“未知词”索引，而不会报错
        src_vocab.set_default_index(src_vocab[self.unk_token])
        print(f'Source vocab size: {len(src_vocab)}')
        
        tgt_vocab = build_vocab_from_iterator(
            self._tokenize(data, self.tgt_tokenize, idx=1),
            min_freq=self.min_freq,
            specials=[self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        )
        tgt_vocab.set_default_index(tgt_vocab[self.unk_token])
        print(f'Target vocab size: {len(tgt_vocab)}')
        
        return src_vocab, tgt_vocab
    
    def convert_text_to_indices(self, text, tokenize, vocab, pad_size):
        """将文本转换为索引序列，包含特殊 token，并进行填充或截断，截断时保留末尾的 <eos>"""
        
        # 讲文本分词为 token，并添加特殊 token
        tokens = [self.sos_token] + tokenize(text) + [self.eos_token]
        
        if len(tokens) < pad_size:
            # 如果长度不足，则填充到指定长度
            tokens = tokens + [self.pad_token] * (pad_size - len(tokens))
        elif len(tokens) > pad_size:
            # 如果长度超过，则截断到指定长度
            # 保证截断后最后一个 token 为 <eos>
            tokens = tokens[:pad_size-1] + [self.eos_token]
        
        # 将 token 转换为索引
        return vocab(tokens)
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        """获取转换后的源和目标索引序列，并对序列进行填充（或截断）到固定长度"""
        src_text, tgt_text = self.raw_data[idx]
        src_indices = self.convert_text_to_indices(src_text, self.src_tokenize, self.src_vocab, self.src_pad_size)
        tgt_indices = self.convert_text_to_indices(tgt_text, self.tgt_tokenize, self.tgt_vocab, self.tgt_pad_size)
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

# 使用示例
if __name__ == '__main__':
    # 加载数据集
    train_raw = Multi30k(split='train', language_pair=('de', 'en'))
    
    # 设置分词器
    tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
    
    # 查看数据集示例
    for de, en in train_raw:
        print(f"源语言: {de}")
        print(f"目标语言: {en}")
        print(f'源语言 tokens: {tokenize_de(de)}')
        print(f'目标语言 tokens: {tokenize_en(en)}')
        break
    
    # 初始化自定义数据集，目标序列填充大小为必填参数（例如固定填充到20），源序列可选择填充
    train_dataset = CustomDataset(train_raw, src_tokenize=tokenize_de, tgt_tokenize=tokenize_en,
                                  min_freq=2, src_pad_size=30, target_pad_size=30)
    
    # 创建 DataLoader，直接传入自定义数据集
    batch_size = 128
    device = torch.device('mps')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 测试一个批次
    for src, tgt in train_dataloader:
        print(f"Source shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Source tensor: {src[0]}")
        print(f"Target tensor: {tgt[0]}")
        break
