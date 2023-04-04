import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(FeedForwardNetwork, self).__init__()
        
        # 定义输入层
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 定义隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # 定义输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 定义激活函数
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 前向传递
        x = self.input_layer(x)
        x = self.activation(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)
        return x


class FeedForwardNetworkFull(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetworkFull, self).__init__()
        
        # 定义两个线性变换层
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 定义激活函数
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 前向传递
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x