
import torch
import string

from transformer_demo import Transformer
from pre_data import get_vocab, MyDataset


# 定义用于预处理文本的函数
def preprocess(text):
    # 将文本转换为小写字母
    text = text.lower()
    # 删除标点符号
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
    # 分割文本为单独的令牌
    tokens = text.split()
    return tokens


def old_main():

    # 定义输入文本
    text = "这是一个解决办法，数据得更全面。"

    with open('three_body.txt', 'r', encoding='utf-8', errors='ignore') as file:  # 'gbk'
        huge_text = file.read()
        vocabulary = dict(get_vocab(huge_text).stoi)
        inverse_dic = {}
        for key,val in vocabulary.items():
            inverse_dic[val] = key
        # print(vocabulary)

        # 定义超参数
        vocab_size = len(vocabulary)
        max_len = 128
        d_model = 512   # dont too large
        num_heads = 16
        d_ff = 20480
        num_layers = 6
        padding_idx = 0

        # 创建 Transformer 模型实例
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer = Transformer(vocab_size, max_len, d_model, num_heads, d_ff, num_layers, padding_idx)
        transformer.to(device)

        # 预处理文本并转换为模型输入所需的格式
        tokens = preprocess(text)
        print(tokens)
        input_seq = torch.zeros((1, max_len), dtype=torch.long)
        for i, token in enumerate(tokens):
            if i >= max_len:
                break
            idx = vocabulary.get(token, vocabulary['<unk>'])
            input_seq[0, i] = idx

        # 传递输入数据给模型进行前向推理
        input_tensor = input_seq.to(device)
        logits = transformer(input_tensor)

        # 打印输出结果的维度
        print(logits.shape)
        print(logits)

        # 获取每个位置上最可能的单词的索引
        predictions = torch.argmax(logits, dim=-1)

        # 打印预测结果及其形状
        for prediction in predictions:
            print(text)
            print(prediction)
            print(prediction.shape)
            print(prediction.tolist())
            print([inverse_dic[x] for x in prediction.tolist()])


def main():
    # 定义超参数
    input_size = 1000
    output_size = 2000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 创建 Transformer 模型
    model = Transformer(input_size, output_size, hidden_size, num_layers, num_heads)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 加载训练和测试数据
    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 开始训练和测试
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        for batch_idx, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()  # 清零梯度
            
            output = model(src, tgt[:-1])  # 预测结果
            loss = criterion(output.reshape(-1, output_size), tgt[1:].reshape(-1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        model.eval()  # 测试模式
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (src, tgt) in enumerate(test_loader):
                output = model(src, tgt[:-1])
                loss = criterion(output.reshape(-1, output_size), tgt[1:].reshape(-1))
                total_loss += loss.item()
            
            avg_loss = total_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_loss:.4f}")
        