import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.Tensor([self.d_k]).to(device=x.device))
        attention_weights = torch.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, v)
        concatenated_attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.output(concatenated_attended_values)

        return output
