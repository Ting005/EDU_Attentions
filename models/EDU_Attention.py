import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.activations import sparsemax
from torch.autograd import Variable

class EmbedAttention(nn.Module):
    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.att_w = nn.Linear(att_size, 1, bias=False)

    def forward(self, mat, len_s):
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pdb.set_trace()
        batch_size, num_edu = x.size()[0], x.size()[1]
        pos = self.pe[:num_edu, :].unsqueeze(dim=0).repeat(batch_size, 1, 1)
        x = x + pos
        return self.dropout(x)


class EduEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden, n_layer):
        super(EduEncoder, self).__init__()
        self.gru = nn.GRU(bidirectional=True, input_size=dim_word, hidden_size=dim_hidden, num_layers=n_layer, batch_first=True)
        self.M1 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M2 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M3 = nn.Parameter(torch.empty((dim_word, dim_hidden * 2)))
        self.M4 = nn.Parameter(torch.empty((dim_hidden * 2, dim_hidden * 2)))

        self.word_attention = EmbedAttention(att_size=dim_hidden * 2)
        self.edu_attention = EmbedAttention(att_size=dim_hidden * 2)
        self.edu_positional_embedding = PositionalEncoding(dim_hidden * 2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.M1, a=math.sqrt(5))
        init.kaiming_uniform_(self.M2, a=math.sqrt(5))
        init.kaiming_uniform_(self.M3, a=math.sqrt(5))
        init.kaiming_uniform_(self.M4, a=math.sqrt(5))

    def _reorder_sent(self, sents, sent_order):
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
        outputs_packed_word_w_asp, _ = self.gru(packed_word_w_asp_embeds)
        hdn_word_w_asp_embeds, _ = pad_packed_sequence(outputs_packed_word_w_asp, batch_first=True)

        alpha = self.word_attention(hdn_word_w_asp_embeds, num_word_seg)
        edu_rep = (hdn_word_w_asp_embeds * alpha).sum(dim=1)
        edu_rep = self._reorder_sent(edu_rep, sent_order)
        alpha = self._reorder_sent(alpha.squeeze(dim=-1), sent_order)
        # add position to edg rep.
        edu_rep = self.edu_positional_embedding(edu_rep)

        # fusion edu and aspect
        # sent_batch_size, max_edu, dim_feature = edu_rep_w_asp.size()
        # aspect_embeds = asp_embed.unsqueeze(dim=0).unsqueeze(dim=0).repeat(sent_batch_size, max_edu, 1)
        # edu_feature = torch.tanh(torch.matmul(edu_rep, self.M4) + torch.matmul(aspect_embeds, self.M3))\
        # beta = self.edu_attention(edu_feature, num_seg, False)
        # without fusion of edu and aspect
        beta = self.edu_attention(edu_rep, num_seg)
        return edu_rep, beta, alpha


class SenEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden, n_layer):
        super(SenEncoder, self).__init__()
        self.gru = nn.GRU(bidirectional=True, input_size=dim_word, hidden_size=dim_hidden, num_layers=n_layer, batch_first=True)
        self.M1 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M2 = nn.Parameter(torch.empty((dim_word, dim_word)))
        self.M3 = nn.Parameter(torch.empty((dim_word, dim_hidden * 2)))
        self.M4 = nn.Parameter(torch.empty((dim_hidden * 2, dim_hidden * 2)))

        self.word_attention = EmbedAttention(att_size=dim_hidden * 2)
        self.edu_attention = EmbedAttention(att_size=dim_hidden * 2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.M1, a=math.sqrt(5))
        init.kaiming_uniform_(self.M2, a=math.sqrt(5))
        init.kaiming_uniform_(self.M3, a=math.sqrt(5))
        init.kaiming_uniform_(self.M4, a=math.sqrt(5))

    def _reorder_sent(self, sents, sent_order):
        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))

        return revs

    def forward(self, sent_word_embed, num_word_sent, asp_embed): #batch_sent, _num_word_sent, all_asp_embs[k, :]
        batch_size, max_words, dim_feature = sent_word_embed.size()
        aspect_embeds = asp_embed.unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size, max_words, 1)
        # fusion: aspect & word
        word_w_asp_embeds = torch.tanh(torch.matmul(sent_word_embed, self.M1) + torch.matmul(aspect_embeds, self.M2))

        lengths_ranked, indices = torch.sort(torch.LongTensor(num_word_sent), dim=0, descending=True)
        _, idx_r = torch.sort(indices, descending=False)
        idx_r = Variable(idx_r).cuda()
        idx_select = Variable(indices).cuda()
        input_ranked = torch.index_select(word_w_asp_embeds, dim=0, index=idx_select)

        input_packed = pack_padded_sequence(input_ranked, lengths=lengths_ranked.cpu().numpy(), batch_first=True)
        output_packed, _ = self.gru(input_packed)
        output_pad, output_len = pad_packed_sequence(output_packed, batch_first=True)
        output_pad_ranked = torch.index_select(output_pad, dim=0, index=idx_r)

        alpha = self.word_attention(output_pad_ranked, num_word_sent)
        ctx_rep = (output_pad_ranked * alpha).sum(dim=1)
        # pdb.set_trace()

        return ctx_rep


class EDU_Attention(nn.Module):

    def __init__(self, dim_word, dim_hidden, n_layer, glove, asp_glove, ASP_DICT, aspect_indexes):
        super(EDU_Attention, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(glove, dtype=torch.float), freeze=False)
        self.asp_embed = nn.Embedding.from_pretrained(torch.tensor(asp_glove, dtype=torch.float), freeze=False)
        self.edu_encoder = EduEncoder(dim_word=dim_word, dim_hidden=dim_hidden, n_layer=n_layer)
        self.sent_encoder = SenEncoder(dim_word=dim_word, dim_hidden=dim_hidden, n_layer=n_layer)

        self.logits = nn.Linear(in_features=dim_hidden * 2, out_features=3, bias=True)
        self.asp_logits = nn.Linear(in_features=dim_hidden * 2, out_features=1, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.1)

        self.linear = nn.Linear(dim_hidden * 4, dim_hidden * 2)
        self.pol = nn.Linear(dim_hidden * 2, 3)
        self.n_label = 3
        self.n_asp = len(ASP_DICT)
        self.ASP_DICT = ASP_DICT
        self.aspect_indexes = aspect_indexes

        ignored_params = list(map(id, self.embed.parameters())) + list(map(id, self.asp_embed.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))

    def _com_orthogonal_reg(self, _batch_max):
        batch_size, num_aspect, num_edu = _batch_max.size()
        scores = torch.matmul(_batch_max, _batch_max.permute(0, 2, 1))
        diag = torch.eye(num_aspect, device=scores.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        orth_loss = torch.sqrt((scores - diag).pow_(2) + 1e-4)
        loss = orth_loss.mean()
        return loss

    def _com_duo_orthogonal_reg(self, _batch_max):
        batch_size, num_aspect, num_edu = _batch_max.size()
        scores_1 = torch.matmul(_batch_max, _batch_max.permute(0, 2, 1))
        diag_1 = torch.eye(num_aspect, device=scores_1.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        orth_loss_1 = torch.sqrt((scores_1 - diag_1).pow_(2) + 1e-4)

        scores_2 = torch.matmul(_batch_max.permute(0, 2, 1), _batch_max)
        diag_2 = torch.eye(num_edu, device=scores_2.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        orth_loss_2 = torch.sqrt((scores_2 - diag_2).pow_(2) + 1e-4)
        loss = orth_loss_1.mean() + orth_loss_2.mean()
        return loss

    def forward(self, _batch_seg, _num_word_seg, _num_seg, _batch_sent, _num_word_sent, _sent_order):
        seg_word_embed = self.embed(_batch_seg)
        seg_word_embed = self.dropout1(seg_word_embed)
        sent_word_embed = self.embed(_batch_sent)
        sent_word_embed = self.dropout1(sent_word_embed)

        all_asp_embs = self.embed(torch.Tensor(self.aspect_indexes).long().to(seg_word_embed.device))

        batch_sent_ctx_rep = []
        batch_edu_reps = []
        batch_beta = []
        batch_alpha = []
        for k in range(self.n_asp):
            edu_reps, beta, alpha = self.edu_encoder(seg_word_embed, _num_word_seg, all_asp_embs[k, :], _num_seg, _sent_order)
            # edu_reps = self.dropout2(edu_reps)
            batch_beta.append(beta.squeeze(dim=-1).unsqueeze(dim=1))  # [batch_size, num_aspect, num_edu]
            batch_alpha.append(alpha.unsqueeze(dim=1))
            batch_edu_reps.append(edu_reps.unsqueeze(dim=1))
            # sentence encoder, ATAE
            ctx_reps = self.sent_encoder(sent_word_embed, _num_word_sent, all_asp_embs[k, :])
            batch_sent_ctx_rep.append(ctx_reps.unsqueeze(dim=1))

        batch_edu_reps = torch.cat(batch_edu_reps, dim=1)
        batch_beta = torch.cat(batch_beta, dim=1)
        batch_alpha = torch.cat(batch_alpha, dim=1)
        _, _, num_edu, _ = batch_alpha.size()
        batch_sent_ctx_rep = torch.cat(batch_sent_ctx_rep, dim=1)
        # var_beta = []
        # for i in range(0, num_edu):
        #     var_beta.append(self._com_orthogonal_reg(batch_alpha[:, :, i, :]).unsqueeze(dim=0))
        # beta_o_reg = torch.cat(var_beta, dim=0).mean()
        batch_sent_rep = (batch_edu_reps * batch_beta.unsqueeze(dim=-1)).sum(dim=2)
        batch_sent_feature = self.linear(torch.cat([batch_sent_ctx_rep, batch_sent_rep], dim=-1))
        batch_sent_feature = self.dropout2(batch_sent_feature)
        # adding GRU for sentence rep
        pol_logits = self.logits(batch_sent_feature)
        asp_logits = self.asp_logits(batch_sent_feature)
        asp_prob = torch.sigmoid(asp_logits.squeeze(dim=-1))
        beta_o_reg = self._com_orthogonal_reg(batch_beta)
        # duo so
        # beta_o_reg = self._com_duo_orthogonal_reg(batch_beta)
        # with alpha involved
        # o_reg = beta_o_reg + alpha_o_reg
        # without alpha involved
        o_reg = beta_o_reg
        return pol_logits, asp_prob, o_reg, [batch_beta, batch_alpha]
