import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out



if __name__ = "__main__":

    # 定义数据和模型参数
    input_dim = 1000
    hidden_dim = 100
    output_dim = 2
    x = torch.randn(32, input_dim)
    y = torch.LongTensor([0, 1] * 16)

    # 初始化模型并进行训练
    model = FFN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, loss.item()))

    # 测试模型
    with torch.no_grad():
        test_x = torch.randn(10, input_dim)
        test_output = model(test_x)
        _, predicted = torch.max(test_output.data, 1)
        print('Predicted Classes:', predicted)
