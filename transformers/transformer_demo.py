import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from ffn import FeedForwardNetworkFull


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward_network = FeedForwardNetworkFull(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attended = self.multi_head_attention(x)
        x = self.layer_norm1(x + attended)
        
        fed_forward = self.feed_forward_network(x)
        x = self.layer_norm2(x + fed_forward)
        
        return x


class TransformerRaw(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, num_layers, padding_idx):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).expand(x.size(0), x.size(1)).long()
        embedded_tokens = self.token_embedding(x)
        x = embedded_tokens + self.positional_encoding[positions]
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        logits = self.output_linear(x)
        
        return logits


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads):
        super(Transformer, self).__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads),
            num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, num_heads),
            num_layers
        )
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt):
        # src: input sequence, shape (seq_len, batch_size)
        # tgt: target sequence, shape (seq_len, batch_size)
        
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)
        
        encoded = self.encoder(src_embedded)
        decoded = self.decoder(tgt_embedded, encoded)
        
        out = self.fc(decoded)
        return out


def test_old():

    # 设置参数
    vocab_size = 1000
    max_len = 50
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 6
    padding_idx = 0

    # 创建Transformer模型
    model = Transformer(vocab_size, max_len, d_model, num_heads, d_ff, num_layers, padding_idx)

    # 创建输入张量
    inputs = torch.LongTensor([[1, 2, 3], [4, 5, 6]])

    # 进行前向传递
    logits = model(inputs)

    # 查看输出的形状
    print(logits.shape)  # torch.Size([2, 3, 1000])

    # 准备输入数据并进行预测
    input_data = ["this is a test", "another test input"]
    tokens = [text.split() for text in input_data]
    token_ids = [[vocab.get(token, 0) for token in sentence] for sentence in tokens]
    padded_inputs = nn.utils.rnn.pad_sequence([torch.LongTensor(sentence) for sentence in token_ids], batch_first=True, padding_value=padding_idx)
    inputs = padded_inputs.to(device)

    logits = model(inputs)

    # 对预测结果进行后处理
    predictions = F.softmax(logits, dim=-1)
    print(predictions)
