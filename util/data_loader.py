from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class CustomDataLoader:
    def __init__(self, src_lang, tgt_lang, src_tokenize, tgt_tokenize, min_freq=2):
        # 设置分词器
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = tgt_tokenize
        
        # 特殊 token
        self.init_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        # 源语言和目标语言
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # 词汇表参数
        self.min_freq = min_freq
        print('Initializing data loader...')

    def _tokenize(self, data_iter, tokenizer, idx, add_special=True):
        """分词函数，返回分词后的结果，可选择是否添加特殊 token
        Args:
            data_iter: 数据迭代器
            tokenizer: 分词器
            idx: 索引，用于在迭代器中选择源语言或目标语言
            add_special: 是否添加特殊 token
        """
        for data in data_iter:
            tokens = tokenizer(data[idx])
            if add_special:
                yield [self.init_token] + tokens + [self.eos_token]
            else:
                yield tokens

    def build_vocab(self, train_iter):
        # 构建源语言词汇表
        # 使用 torchtext 的 build_vocab_from_iterator 函数构建源语言的词汇表
        # 参数解释：
        #   - 第一个参数是一个生成器，通过 _tokenize 方法遍历训练集中的文本数据并进行分词，其中 idx=0 表示选择数据中的第一个元素（源语言句子）。
        #   - min_freq 用于设置词汇的最小词频，词频低于该值的词将不会进入词汇表
        #   - specials 列表定义了一些特殊的 token，例如填充符 (<pad>)、未知词 (<unk>)、句子开始 (<sos>) 和句子结束 (<eos>)
        src_vocab = build_vocab_from_iterator(
            self._tokenize(train_iter, self.src_tokenize, idx=0),
            min_freq=self.min_freq,
            specials=[self.pad_token, self.unk_token, self.init_token, self.eos_token]
        )
        
        # 设置默认索引：当词汇表中不存在某个 token 时，返回默认索引
        # 这样可以保证在文本转换为数值序列时，如果遇到未收录的词，会返回一个预设的“未知词”索引，而不会报错
        src_vocab.set_default_index(src_vocab[self.unk_token])
        print(f'Source vocab size: {len(src_vocab)}')
        
        # 构建目标语言词汇表
        tgt_vocab = build_vocab_from_iterator(
            self._tokenize(train_iter, self.tgt_tokenize, idx=1),
            min_freq=self.min_freq,
            specials=[self.pad_token, self.unk_token, self.init_token, self.eos_token]
        )
        tgt_vocab.set_default_index(tgt_vocab[self.unk_token])
        print(f'Target vocab size: {len(tgt_vocab)}')
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        return src_vocab, tgt_vocab

    def _sequential_transform(self, tokenizer, vocab):
        """将文本转换为数值序列的函数"""
        def transform(text):
            # 将句子分词为token，并在首尾添加特殊 token
            tokens = [self.init_token] + tokenizer(text) + [self.eos_token]
            
            # 将 token 转换为索引
            return vocab(tokens)
        
        return transform

    def get_transform(self):
        """获取源语言和目标语言的转换函数"""
        src_transform = self._sequential_transform(self.src_tokenize, self.src_vocab)
        tgt_transform = self._sequential_transform(self.tgt_tokenize, self.tgt_vocab)
        return src_transform, tgt_transform

    def collate_fn(self, batch, device=torch.device('cpu')):
        """批处理函数，处理填充和维度转换"""
        src_batch, tgt_batch = [], []
        for sample in batch:
            src = torch.tensor(sample[0], dtype=torch.long)
            tgt = torch.tensor(sample[1], dtype=torch.long)
            src_batch.append(src)
            tgt_batch.append(tgt)
        
        # 填充序列
        src_batch = pad_sequence(src_batch, padding_value=self.src_vocab[self.pad_token])
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.tgt_vocab[self.pad_token])
        
        return src_batch.to(device), tgt_batch.to(device)

    def make_iter(self, dataset, batch_size=128, device=torch.device('cpu')):
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate_fn(batch, device)
        )

# 使用示例
if __name__ == '__main__':
    # 加载数据集
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    valid_iter = Multi30k(split='valid', language_pair=('de', 'en'))
    
    # 设置分词器
    tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
    
    # 查看数据集
    for de, en in train_iter:
        print(f"German: {de}")
        print(f"English: {en}")
        print(f'German tokens: {tokenize_de(de)}')
        print(f'English tokens: {tokenize_en(en)}')
        break
    
    # 初始化数据加载器
    loader = CustomDataLoader(src_lang='de', tgt_lang='en', src_tokenize=tokenize_de, tgt_tokenize=tokenize_en, min_freq=2)
    
    # 构建词汇表（需要完整遍历训练集）
    src_vocab, tgt_vocab = loader.build_vocab(train_iter)
    print('一些token对应的索引：', tgt_vocab['hello'], tgt_vocab['<eos>'])
    
    # 获取转换函数
    src_transform, tgt_transform = loader.get_transform()
    print('句子转换为索引序列：', tgt_transform('hello world!'))
    
    # 重新加载带转换的数据集（需要将原始文本转换为索引）
    train_dataset = [(src_transform(de), tgt_transform(en)) for de, en in Multi30k(split='train', language_pair=('de', 'en'))]
    valid_dataset = [(src_transform(de), tgt_transform(en)) for de, en in Multi30k(split='valid', language_pair=('de', 'en'))]
    
    # 创建数据加载器
    train_dataloader = loader.make_iter(train_dataset, batch_size=128, device=torch.device('mps'))
    valid_dataloader = loader.make_iter(valid_dataset, batch_size=128, device=torch.device('mps'))
    
    # 测试一个批次
    for src, tgt in train_dataloader:
        print(f"Source shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        break
