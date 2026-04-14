import torch
from torch import nn
import torch.nn.functional as F


#做的是将词id转化为向量(vocab_size为词汇表大小，d_model为Token词向量长度)
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
    

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, Maxlen, dropout, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, Maxlen, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x)
        x = self.dropout(x)
        return x






#d_model为Token词向量长度，Transformer 把“每个 token”当作一个独立样本，layernorm的作用是对每个样本进行归一化，保证每个样本的均值为0，方差为1，增强模型的稳定性和收敛速度
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

#前馈网络，输入输出维度都是d_model，hidden为前馈网络的隐藏层维度, 通常 4×d_model，dropout为dropout率
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

