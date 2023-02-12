import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.entmax.activations import sparsemax, entmax15
from torch.autograd import Variable

class EmbedAttention(nn.Module):

    def _masked_softmax(self, mat, len_s):
        len_s = len_s.type_as(mat.data)  # .long()
        idxes = torch.arange(0, int(len_s.max().item()), out=mat.data.new(int(len_s.max().item())).long()).unsqueeze(1)
        mask = (idxes.float() < len_s.unsqueeze(0)).float()
        mask = mask.t()
        mat_v, _ = mat.max(dim=-1)
        mat = mat - mat_v.unsqueeze(dim=-1)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(1, True) + 1e-4
        return exp / sum_exp.expand_as(exp)

    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.att_w = nn.Linear(att_size, 1, bias=False)

    def forward(self, mat, len_s, is_word=True):
        att = self.att_w(mat).squeeze(-1)
        len_s = len_s.type_as(mat.data)
        idxes = torch.arange(0, int(len_s.max().item()), out=mat.data.new(int(len_s.max().item())).long()).unsqueeze(1)
        rev_mask = (idxes.float() >= len_s.unsqueeze(0)).bool()
        rev_mask = rev_mask.t()
        att[rev_mask] = -float("inf")
        out = sparsemax(att, dim=1).unsqueeze(-1)  #
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pdb.set_trace()
        batch_size, num_edu = x.size()[0], x.size()[1]
        pos = self.pe[:num_edu, :].unsqueeze(dim=0).repeat(batch_size, 1, 1)
        # x = x + self.pe[:x.size(0), :]
        x = x + pos
        # return self.dropout(x)
        return x


class EduEncoder_W_ASP(nn.Module):
    def __init__(self, dim_word, dim_hidden, n_layer):
        super(EduEncoder_W_ASP, self).__init__()
        self.lstm1 = nn.GRU(bidirectional=True, input_size=dim_word, hidden_size=dim_hidden, num_layers=n_layer, batch_first=True)
        # self.sent_encoder = nn.GRU(bidirectional=True, input_size=dim_hidden*2, hidden_size=dim_hidden, num_layers=n_layer, batch_first=True)
        # self.lstm1 = nn.LSTM(bidirectional=True, input_size=dim_word, hidden_size=dim_hidden, num_layers=n_layer, batch_first=True)
        # self.linear1 = nn.Linear(in_features=dim_word * 2, out_features=dim_word, bias=False)
        # self.linear2 = nn.Linear(in_features=dim_hidden * 2 + dim_word, out_features=dim_hidden*2, bias=False)
        self.M1 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M2 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M3 = nn.Parameter(torch.empty((dim_word, dim_hidden*2)))
        self.M4 = nn.Parameter(torch.empty((dim_hidden*2, dim_hidden*2)))

        self.word_attention = EmbedAttention(att_size=dim_hidden*2)
        self.edu_attention = EmbedAttention(att_size=dim_hidden*2)
        self.edu_positional_embedding = PositionalEncoding(dim_hidden*2)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.M1, a=math.sqrt(5))
        init.kaiming_uniform_(self.M2, a=math.sqrt(5))
        init.kaiming_uniform_(self.M3, a=math.sqrt(5))
        init.kaiming_uniform_(self.M4, a=math.sqrt(5))

    def  _reorder_sent(self, sents, sent_order):
        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))

        return revs

    def forward(self, word_embeds, num_word_seg, asp_embed, num_seg, sent_order):
        edu_batch_size, max_words, dim_feature = word_embeds.size()

        aspect_embeds = asp_embed.unsqueeze(dim=0).unsqueeze(dim=0).repeat(edu_batch_size, max_words, 1)
        # fusion: aspect & word
        word_w_asp_embeds = torch.tanh(torch.matmul(word_embeds, self.M1) + torch.matmul(aspect_embeds, self.M2))

        packed_word_w_asp_embeds = pack_padded_sequence(word_w_asp_embeds, num_word_seg, batch_first=True)
        outputs_packed_word_w_asp, _ = self.lstm1(packed_word_w_asp_embeds)
        hdn_word_w_asp_embeds, _ = pad_packed_sequence(outputs_packed_word_w_asp, batch_first=True)

        word_att_score_w_asp = self.word_attention(hdn_word_w_asp_embeds, num_word_seg, True)
        edu_rep_w_asp = (hdn_word_w_asp_embeds * word_att_score_w_asp).sum(dim=1)
        edu_rep_w_asp = self._reorder_sent(edu_rep_w_asp, sent_order)
        word_att_score_w_asp = self._reorder_sent(word_att_score_w_asp.squeeze(dim=-1), sent_order)
        # add position to edg rep.
        edu_rep_w_asp = self.edu_positional_embedding(edu_rep_w_asp)
        sent_batch_size, max_edu, dim_feature = edu_rep_w_asp.size()
        aspect_embeds = asp_embed.unsqueeze(dim=0).unsqueeze(dim=0).repeat(sent_batch_size, max_edu, 1)

        # fusion edu and aspect
        edu_w_asp_embeds = torch.tanh(torch.matmul(edu_rep_w_asp, self.M4) + torch.matmul(aspect_embeds, self.M3))
        alpha = self.edu_attention(edu_w_asp_embeds, num_seg, False)
        return edu_rep_w_asp, alpha, word_att_score_w_asp


class HANREG(nn.Module):

    def __init__(self, dim_word, dim_hidden, n_layer, glove, asp_glove, embed_dropout, ASP_DICT, aspect_indexes):
        super(HANREG, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(glove, dtype=torch.float), freeze=False)
        self.asp_embed = nn.Embedding.from_pretrained(torch.tensor(asp_glove, dtype=torch.float), freeze=False)

        self.edu_encoder_w_asp = EduEncoder_W_ASP(dim_word=dim_word, dim_hidden=dim_hidden, n_layer=n_layer)
        # self.edu_encoder = EduEncoder(dim_word=dim_word, dim_hidden=dim_hidden, n_layer=n_layer, n_asp=n_aspect)

        self.logits = nn.Linear(in_features=dim_hidden * 2, out_features=3, bias=True)
        self.asp_logits = nn.Linear(in_features=dim_hidden * 2, out_features=1, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.1)

        self.pol = nn.Linear(dim_hidden * 2, 3)
        self.n_label = 3
        self.n_asp = len(ASP_DICT)
        self.ASP_DICT = ASP_DICT
        self.aspect_indexes = aspect_indexes

        ignored_params = list(map(id, self.embed.parameters())) + list(map(id, self.asp_embed.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))

    def _com_orthogonal_reg(self, _batch_max):
        # [batch_size, num_aspect, num_edu]
        batch_size, num_aspect, num_edu = _batch_max.size()
        scores = torch.matmul(_batch_max, _batch_max.permute(0, 2, 1))
        diag = torch.eye(num_aspect, device=scores.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        orth_loss = torch.sqrt((scores - diag).pow_(2) + 1e-4)
        # orth_loss = torch.norm((scores - diag), p=1)

        # pdb.set_trace()
        loss = orth_loss.mean()

        return loss

    def forward(self, _batch_seg, _num_word_seg, _num_seg, _sent_order, _num_asp):
        word_embed = self.embed(_batch_seg)
        word_embed = self.dropout1(word_embed)
        all_asp_embs = self.embed(torch.Tensor(self.aspect_indexes).long().to(word_embed.device))
        # all_asp_embs = self.dropout2(all_asp_embs)

        batch_edu_w_asp_embeds = []
        batch_alpha = []
        batch_asp_word_attention = []
        for k in range(self.n_asp):
            edu_w_asp_embeds, alpha, word_att_score_w_asp = self.edu_encoder_w_asp(word_embed, _num_word_seg, all_asp_embs[k, :], _num_seg, _sent_order)
            edu_w_asp_embeds = self.dropout2(edu_w_asp_embeds)

            batch_alpha.append(alpha.squeeze(dim=-1).unsqueeze(dim=1)) # [batch_size, num_aspect, num_edu]
            batch_asp_word_attention.append(word_att_score_w_asp.unsqueeze(dim=1))
            batch_edu_w_asp_embeds.append(edu_w_asp_embeds.unsqueeze(dim=1))

        # pdb.set_trace()
        batch_edu_w_asp_embeds = torch.cat(batch_edu_w_asp_embeds, dim=1)
        batch_alpha = torch.cat(batch_alpha, dim=1)
        batch_asp_word_attention = torch.cat(batch_asp_word_attention, dim=1)
        _, _, num_edu, _ = batch_asp_word_attention.size()
        var_beta = []
        for i in range(0, num_edu):
            var_beta.append(self._com_orthogonal_reg(batch_asp_word_attention[:, :, i, :]).unsqueeze(dim=0))
        beta_o_reg = torch.cat(var_beta, dim=0).mean()
        # to compute beta
        # batch_beta = self.edu_encoder(word_embed, _num_word_seg, _num_seg, _sent_order)
        # compute by using self-attention
        # batch_likelihood = batch_alpha * (batch_beta.permute(0, 2, 1))
        batch_sent_rep = (batch_edu_w_asp_embeds * batch_alpha.unsqueeze(dim=-1)).sum(dim=2)
        # pdb.set_trace()
        pol_logits = self.logits(batch_sent_rep)
        asp_logits = self.asp_logits(batch_sent_rep)
        asp_prob = torch.sigmoid(asp_logits.squeeze(dim=-1))
        # pdb.set_trace()
        # alpha_o_reg = self._com_orthogonal_reg(batch_alpha, _num_seg, _num_asp, _for_asp=True)
        alpha_o_reg = self._com_orthogonal_reg(batch_alpha)
        # beta_o_reg = self._com_orthogonal_reg(batch_beta, _num_seg, _num_asp, _for_asp=True)
        # pdb.set_trace()
        o_reg = beta_o_reg +  alpha_o_reg
        return pol_logits, asp_prob, o_reg, [batch_alpha, batch_asp_word_attention]# + beta_o_reg
