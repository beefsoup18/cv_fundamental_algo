import traceback
import time
import os

import torch
import string
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_demo import Transformer
from pre_data import get_vocab, MyDataset, split_chinese


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


def single_train(src, tgt):  # model, optimizer, criterion, 
    optimizer.zero_grad()  # 清零梯度
    src = src.to(device)
    tgt = tgt.to(device)
    try:

        output = model(src, tgt)  # 预测结果    #tgt[:-1]
    except:
        print("output = model(src, tgt)  FAILED !!!")
        # print("output", output)
        # print("output", output.reshape(-1, output_size))
        print("src", src)
        print("tgt", tgt)
        traceback.print_exc()
    # loss = criterion(output.reshape(-1, output_size), tgt.reshape(-1))  # 计算损失   #tgt[1:]
    loss = criterion(output, tgt)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss


def single_test(src, tgt, total_loss):  # model, 
    src = src.to(device)
    tgt = tgt.to(device)
    output = model(src, tgt)  # tgt[:-1]
    loss = criterion(output, tgt)
    total_loss += loss.item()
    return total_loss, torch.argmax(output, dim=-1)


def batch_conduct(batch_size, train_text, test_text):
    print("len(train_text):", len(train_text))
    train_dataset = MyDataset(train_text[:-1], train_text[1:], vocabulary)
    test_dataset = MyDataset(test_text[:-1], test_text[1:], vocabulary)

    # 参数 batch_size 指定了每个 batch 中包含的样本数量
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 开始训练和测试
    for epoch in range(num_epochs):
        now = time.time()

        # 训练模式
        model.train()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            loss = single_train(src, tgt)
            if batch_idx % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}], "\
                        f"Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}", 
                        f"takes time {time.time() - now}")
                now = time.time()
        
        # 测试模式
        prediction = ""
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (src, tgt) in enumerate(test_loader):
                total_loss, output = single_test(src, tgt, total_loss)
                # print("tgt",len(tgt))
                # print(tgt)
                # print("output",len(output))
                # print(output)
                prediction += "".join([inverse_vocabulary[x] for x in output.tolist()])
            avg_loss = total_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_loss:.4f}", time.time())
            if epoch == num_epochs - 1:
                print(test_text)
                print(prediction)
            elif epoch % 10 == 0:
                print(prediction)


def single_conduct(train_text, test_text):  # model, optimizer, criterion, vocabulary, 

    train_dataset = MyDataset(train_text[:-1], train_text[1:], vocabulary)
    test_dataset = MyDataset(test_text[:-1], test_text[1:], vocabulary)
    train_loader = train_dataset
    test_loader = test_dataset

    # 开始训练和测试
    for epoch in range(num_epochs):

        # 训练模式
        model.train()
        # train_loader = train_loader.to(device)
        for idx, (src, tgt) in enumerate(train_loader):
            single_train(src, tgt)
        
        # 测试模式
        model.eval()
        prediction = ""
        with torch.no_grad():
            total_loss = 0
            # test_loader = test_loader.to(device)
            for idx, (src, tgt) in enumerate(test_loader):
                total_loss, output = single_test(src, tgt, total_loss)
                prediction += inverse_vocabulary[output.tolist()[0]]
            avg_loss = total_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_loss:.4f}")
            if epoch == num_epochs - 1:
                print(test_text)
                print(prediction)



if __name__ == "__main__":

    try:
        os.mkdir("model")
    except:
        pass

    vocab_file = "three_body3.txt"
    train_file = "three_body3.txt"
    test_file = "text.txt"

    with open(vocab_file, 'r', encoding='utf-8', errors='ignore') as file:  # 'gbk'
        huge_text = file.read()
        vocabulary = dict(get_vocab(huge_text).stoi)
        inverse_vocabulary = {}
        for key,val in vocabulary.items():
            inverse_vocabulary[val] = key
        # print(inverse_vocabulary)

        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = [split_chinese(line.strip()) for line in f]
            train_text = [char for line in train_text for char in line]
            train_text = list("".join(train_text).replace("\u3000", ""))
        with open(test_file, 'r', encoding='utf-8') as f:
            test_text = [split_chinese(line.strip()) for line in f]
            test_text = [char for line in test_text for char in line]
            test_text = list("".join(test_text).replace("\u3000", ""))

    # 定义超参数
    # input_size = 1000
    # output_size = 2000
    hidden_size = 512
    num_layers = 5
    num_heads = 16
    batch_size = 320
    num_epochs = 200
    learning_rate = 0.001

    print("len_vocab", len(vocabulary))
    input_size = len(vocabulary)
    output_size = len(vocabulary)

    # 创建 Transformer 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(input_size, output_size, hidden_size, num_layers, num_heads)
    model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.functional.cross_entropy

    # single_conduct(train_text[0], test_text[0])

    batch_conduct(batch_size, train_text, test_text)

    model.save('model')