import torch
import torch.nn as nn
from torchtext.vocab import GloVe
from transformers import BertTokenizer


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        
        # 加载预训练的GloVe嵌入向量
        self.embedding = GloVe(name='6B', dim=embedding_dim)
        
        # 定义神经网络层
        self.fc1 = nn.Linear(embedding_dim*vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x).view(1, -1)
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


if __name__ = "__main__":
    
    # 定义数据和模型参数
    vocab_size = 10000
    embedding_dim = 300
    hidden_dim = 100
    output_dim = 2
    #x = torch.LongTensor([[1, 2, 3, 4, 5]])   # 输入的单词ID序列

    with open('text.txt', 'r') as file:
        text = file.read()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 使用分词器对文本进行分词，并将每个词转换为对应的整数ID
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # 将整数ID组成的列表转换为PyTorch张量
    x = torch.LongTensor([ids])


    # 初始化模型并进行训练
    model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, torch.LongTensor([0]))
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, loss.item()))

    # 测试模型
    with torch.no_grad():
        test_x = torch.LongTensor([[6, 7, 8, 9, 10]])
        test_output = model(test_x)
        _, predicted = torch.max(test_output.data, 1)
        print('Predicted Classes:', predicted)
