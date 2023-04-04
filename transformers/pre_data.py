import torch
from torch.utils.data import Dataset

from torchtext.vocab import Vocab
from collections import Counter


def get_vocab(train_texts, train_labels=[0, 1]):
    # 创建计数器并统计所有令牌的出现次数
    counter = Counter()
    for text in train_texts:
        counter.update(text.split())

    # 创建词汇表并加载计数器以获取前k个最常见的令牌
    vocab = Vocab(counter, max_size=100000, specials=['<pad>', '<unk>'])  # max_size=10000, specials=['<pad>', '<unk>']
    return vocab


def test_vocab():
    # 定义训练文本和标签
    train_texts = ['this is a sentence', 'this is another sentence']
    vocabulary = dict(get_vocab(train_texts).stoi)
    # 打印词汇表的大小
    print(len(vocab))

def split_chinese(text):
	text = ' '.join(text)
	return text.split(" ")

class MyDataset(Dataset):
    def __init__(self, train_file, test_file, vocab):
        # 从文件中读取源和目标文本
        with open(train_file, 'r', encoding='utf-8') as f:
            self.src = [line.strip() for line in f]
        with open(test_file, 'r', encoding='utf-8') as f:
            self.tgt = [line.strip() for line in f]
        
        # 将源和目标文本转换为 PyTorch 张量
        self.src_tensor = []
        self.tgt_tensor = []
        for i in range(len(self.src)):
            src_words = split_chinese(self.src[i])  # self.src[i].split()
            tgt_words = split_chinese(self.tgt[i])  #self.tgt[i].split()
            # print(split_chinese(self.src[i]))
            # print(src_words.split())
            
            src_tensor = torch.tensor([vocab[word] for word in src_words], dtype=torch.long)
            tgt_tensor = torch.tensor([vocab[word] for word in tgt_words], dtype=torch.long)
            
            self.src_tensor.append(src_tensor)
            self.tgt_tensor.append(tgt_tensor)
            print(len(src_tensor))
        
        self.num_samples = len(self.src)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.src_tensor[index], self.tgt_tensor[index]



with open('three_body.txt', 'r', encoding='utf-8', errors='ignore') as file:  # 'gbk'
    huge_text = file.read()
    vocabulary = dict(get_vocab(huge_text).stoi)



if __name__ == "__main__":
	obj = MyDataset("text.txt", "text.txt", vocabulary)
