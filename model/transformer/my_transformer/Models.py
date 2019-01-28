import torch
import torch.nn as nn
import numpy as np
import Constants
from Layers import EncoderLayer, DecoderLayer

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # dim == 2 ???
    # ne compute seq != PAD, same is 0, not same is 1.
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinsoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        # hid_idx //2 ???
        return position / np.power(10000,2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position,hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    # eq, same is 1, not same is 0.
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """  FOr masking out the subsequent info. """
    """
    Below the attention mask shows the position each tgt word (row) is allowed to look at (column). 
    Words are blocked for attending to future words during training.
    """

    sz_b, len_s = seq.size()
    # triu 转换成上三角矩阵， diagonal=1代表矩阵对角线为0.
    subsequent_mask = torch.triu(
        torch.ones((len_s,len_s), device=seq.devoce, dtype=torch.uint8),diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, len_max_seq, d_word_vec,n_layers,
                  n_head, d_k, d_v, d_model,d_inner, dropout=0.1):
        super(Encoder,self).__init__()
        n_position = len_max_seq + 1

        # word2vec, glove 使用
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinsoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attn=False):
        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask()

















































