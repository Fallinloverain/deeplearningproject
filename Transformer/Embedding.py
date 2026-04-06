import torch
from torch import nn
import torch.nn.functional as F


#做的是将词id转化为向量(vocab_size为词汇表大小，d_model为词向量长度)
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=1) #padding 是要 embedding[1] = 固定为 0（或不更新）
        
#创建位置编码表， Maxlen为最大支持的句子长度
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, Maxlen, device):
        super(PositionalEmbedding, self).__init__()
        #创建一个矩阵：[Maxlen 行，每行 d_model 维]
        self.encoding = torch.zeros(Maxlen, d_model, device)
        self.encoding.requires_grad_(False)

        #构造位置 pos ：[Maxlen × 1]
        pos = torch.arange(0, Maxlen, device)
        pos = pos.float().unsqueeze(1)

        #构造维度索引 2i ：[1 × (d_model/2)]
        _2i = torch.arange(0, d_model, 2, device)

        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]