from numpy import zeros
from torch.nn.init import normal_
import torch.nn as nn
import torch


class FLASHAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device, expand=2, c=256, s=128, src_len=512, dropout=0, bias=True):
        super().__init__()
        """
        :param embed_dim    词嵌入的维度，论文中的d
        :param num_heads:   多头注意力中注意力的头数
        """
        self.e = expand * embed_dim
        self.d = embed_dim
        self.c = c
        # 本地注意力的长度
        self.g = src_len // c
        # 整个序列被划分为(src_len // c)个本地块
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.bias = torch.normal(mean=torch.zeros(size=(c,)), std=0.02).to(device)
        self.bias.requires_grad = True
        self.gamma = torch.normal(mean=torch.zeros(size=(4, s)), std=0.02).to(device)
        self.gamma.requires_grad = True
        self.beta = torch.normal(mean=torch.zeros(size=(4, s)), std=0.02).to(device)
        self.beta.requires_grad = True
        self.dense_xs = nn.Linear(self.d, s)
        self.dense_ve = nn.Linear(self.d, self.e)
        self.dense_ue = nn.Linear(self.d, self.e)
        self.dense_od = nn.Linear(self.e, self.d)
        self.act = torch.tanh

        self.zeros = torch.zeros(size=(src_len, src_len))

        assert self.head_dim * num_heads == embed_dim
        assert self.c * self.g == src_len

        for p in self.parameters():
            normal_(p, mean=0.0, std=0.02)
    
    def _global_linear_attn(self, q, k, v):
        """
        q, k:   [b, G, C, s]
        v:      [b, G, C, e]
        q(kv) 的计算复杂度是线性的
        """
        kv = torch.einsum("bgcs,bgce->bse", [k, v])
        return torch.einsum("bgcs,bse->bgce", [q, kv])

    def _local_quadratic_attn(self, q, k, v):
        """
        q, k:   [b, G, C, s]
        v:      [b, G, C, e]
        本地的(qk)v复杂度是平方的
        """
        qk = torch.einsum("bgns,bgms->bgnm", [q, k])
        a = torch.relu(qk + self.bias) ** 2
        # a = torch.relu(qk) ** 2
        return torch.einsum("bgnm,bgme->bgne", [a, v])

    def attn(self, x, v):
        """
        x:  [batch_size, G, C, embed_dim(d)]
        v:  [batch_size, G, C, e]
        """
        z = self.dense_xs(x)
        z = self.act(z)                                         # [b, g, c, s]
        q_quad = z * self.gamma[0, :] + self.beta[0, :]
        k_quad = z * self.gamma[1, :] + self.beta[1, :]
        q_lin = z * self.gamma[2, :] + self.beta[2, :]
        k_lin = z * self.gamma[3, :] + self.beta[3, :]
        v_quad = self._local_quadratic_attn(q_quad, k_quad, v)
        v_lin = self._global_linear_attn(q_lin, k_lin, v)
        return v_quad + v_lin                                   # [b, g, c, e]

    def forward(self, query, key, value: torch.Tensor, attn_mask=None, key_padding_mask=None):
        # 调用方法：self.self(hidden_states, hidden_states,hidden_states,attn_mask = None,key_padding_mask = attention_mask)
        # hidden_states: [src_len, batch_size, hidden_size]

        x = value.transpose(0, 1)                               # [batch_size, src_len, hidden_size(d)]
        x = x.reshape(x.shape[0], self.g, self.c, x.shape[2])   # [b, g, c, d]
        v = self.dense_ve(x)                                    # [b, g, c, e]
        v = self.act(v)
        v = self.attn(x, v)                                     # [b, g, c, e]
        
        # 相当于获取query和key
        
        u = self.dense_ue(x)
        u = self.act(u)                                         # [b, g, c, e]
        
        # 相当于获取value
        
        o = self.dense_od(u * v)                                # [b, g, c, d]
        o = o.reshape(o.shape[0], o.shape[1] * o.shape[2], o.shape[3])
        o = o.transpose(0, 1)
        
        # 后续处理不需要在这里做完
        # 第二个返回值是注意力权重，但是模型里只需要输出就够了，就懒得弄了
        
        return o, self.zeros
