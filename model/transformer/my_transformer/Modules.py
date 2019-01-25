import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot - Product Attention """
    def __init__(self, root_dk, attn_dropout=0.1):
        # Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx :
        super(ScaledDotProductAttention,self).__init__()
        self.root_dk = root_dk
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)  # dim = 2 ???

    def forward(self, q, k, v, mask=None):
        attn_sco = torch.bmm(q, k.transpose(1,2))
        attn_sco_div = attn_sco / self.root_dk

        if mask is not None:
            attn_sco_div = attn_sco_div.masked_fill(mask, -np.inf)
            # Fills elements of self tensor with value where mask is one.
        attn_sco_div = self.softmax(attn_sco_div)
        attn_sco_div = self.dropout(attn_sco_div)
        output = torch.bmm(attn_sco_div, v)

        return output, attn_sco_div

