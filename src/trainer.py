import pdb

import torch.nn.functional as F


def train_single_label_wo_asp(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels, batch_text = _inputs
    batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels = [v.to(_compute_device) for v in [batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels]]
    pol_logits, scores = _model(batch_sent, num_sent_words.cpu())
    # pdb.set_trace()
    loss = F.cross_entropy(pol_logits, batch_pol_labels)
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()


def train_single_label(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels, batch_text = _inputs

    batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels = [v.to(_compute_device)
                                                                     for v in [batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels]]
    # pol_logits, _ = _model(batch_sent, num_sent_words.cpu())
    pol_logits, _ = _model(batch_sent, num_sent_words.cpu(), batch_asp_idxes)
    # pdb.set_trace()
    loss = F.cross_entropy(pol_logits, batch_pol_labels)
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()


def train_han_single_label(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_seg, segs, num_word_seg, num_seg, sent_order, batch_seg_asp, batch_pol_labels = _inputs

    batch_seg, sent_order, batch_seg_asp, batch_pol_labels = [v.to(_compute_device)
                                                                               for v in [batch_seg, sent_order, batch_seg_asp, batch_pol_labels]]
    # pol_logits, _ = _model(batch_sent, num_sent_words.cpu())
    pol_logits, _, _ = _model(batch_seg, num_word_seg, num_seg, sent_order, batch_seg_asp)
    # pdb.set_trace()
    loss = F.cross_entropy(pol_logits, batch_pol_labels)
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()


def train_single_label_gac(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = _inputs
    batch_sent, batch_asp_labels, batch_pol_labels = [v.to(_compute_device) for v in [batch_sent, batch_asp_labels, batch_pol_labels]]
    # pdb.set_trace()
    prob_sentiment, x, y = _model(batch_sent, batch_asp_labels)
    loss = F.cross_entropy(prob_sentiment, batch_pol_labels)
    # pdb.set_trace()
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()


def train_single_bert(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, sent_masks, batch_pol_labels = [v.to(_compute_device) for v in _inputs]
    # pdb.set_trace()
    pol_logits = _model(batch_sent, sent_masks)
    loss = F.cross_entropy(pol_logits, batch_pol_labels)
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()


def train_multilabel_han(_compute_device, _model, _optimizer, _inputs, _total_num_asp):
    _model.train()
    batch_seg, segs, num_word_seg, num_seg, sent_order, batch_asplabels, num_asp, batch_pol_labels = _inputs
    batch_seg, sent_order, num_asp, batch_pol_labels, batch_asplabels = [v.to(_compute_device) for v in [batch_seg, sent_order, num_asp, batch_pol_labels, batch_asplabels]]
    # pdb.set_trace()
    pol_logits, asp_prob, var, _ = _model(batch_seg, num_word_seg, num_seg, sent_order, num_asp)
    pol_loss = F.cross_entropy(pol_logits.permute(0, 2, 1), batch_pol_labels, ignore_index=-1)
    loss_asp = F.binary_cross_entropy(asp_prob, batch_asplabels)
    # good for rest & laptop
    # loss = pol_loss + (1 / _total_num_asp) * loss_asp + 0.1 * var
    # loss = 0.7 * pol_loss + 0.2 * loss_asp + 0.1 * var
    # set for mams
    # loss = 0.8 * pol_loss + 0.1 * loss_asp + 0.1 * var
    # set for rest
    # loss = 0.45 * pol_loss + 0.40 * loss_asp + 0.15 * var
    # loss = pol_loss + loss_asp + 0.15 * var
    # loss = 0.60 * pol_loss + 0.40 * loss_asp
    # set for latpop
    # loss = 0.4 * pol_loss + 0.5 * loss_asp + 0.10 * var
    # loss = 0.8 * pol_loss + 0.2 * loss_asp + 0.1 * var
    # loss = pol_loss + loss_asp + 0.1 * var  # 0.1 * + loss_asp#
    loss = pol_loss + loss_asp + 0.1 * var
    # loss = 0.5 * pol_loss + 0.4 * loss_asp + 0.10 * var

    # pdb.set_trace()
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()

    return loss.data.cpu().numpy()


def train_multilabel_flat(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = _inputs
    batch_sent, batch_asp_labels, batch_pol_labels = [v.to(_compute_device) for v in [batch_sent, batch_asp_labels, batch_pol_labels]]
    prob_asp, prob_sentiment, attn_asp, attn_sen = _model(batch_sent, num_sent_words)
    # prob_asp, prob_sentiment = _model(batch_sent, num_sent_words)

    pol_loss = F.cross_entropy(prob_sentiment.permute(0, 2, 1), batch_pol_labels, ignore_index=-1)
    loss_asp = F.binary_cross_entropy(prob_asp, batch_asp_labels)
    loss = 0.5 * pol_loss + 0.5 * loss_asp
    # pdb.set_trace()
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()

def train_multilabel_flat_can(_compute_device, _model, _optimizer, _inputs):
    _model.train()
    batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = _inputs
    batch_sent, batch_asp_labels, batch_pol_labels = [v.to(_compute_device) for v in [batch_sent, batch_asp_labels, batch_pol_labels]]
    prob_asp, prob_sentiment, o_reg = _model(batch_sent, num_sent_words)
    # prob_asp, prob_sentiment = _model(batch_sent, num_sent_words)

    pol_loss = F.cross_entropy(prob_sentiment.permute(0, 2, 1), batch_pol_labels, ignore_index=-1)
    loss_asp = F.binary_cross_entropy(prob_asp, batch_asp_labels)
    loss = pol_loss + 0.2 * loss_asp + 0.1 * o_reg
    # pdb.set_trace()
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy()

def train_multilabel_han_bert(_compute_device, _model, _optimizer, _inputs, _total_num_asp, asp_ids_and_mask=None):
    _model.train()
    [seg_input_ids, seg_attention_mask],  num_segs, asp_labels, pol_labels = _inputs
    # pdb.set_trace()
    asp_ids, asp_ids_mask = asp_ids_and_mask
    # pdb.set_trace()
    seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels = \
        [v.to(_compute_device) for v in [seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels]]

    pol_logits, asp_prob, var = _model(num_segs, seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask)

    loss_asp = F.binary_cross_entropy(asp_prob, asp_labels)
    pol_loss = F.cross_entropy(pol_logits.permute(0, 2, 1), pol_labels, ignore_index=-1)

    # loss = pol_loss + (1 / _total_num_asp) * loss_asp + 0.1 * var
    # loss = pol_loss + loss_asp + 0.1 * var
    # loss = 0.7 * pol_loss + 0.1 * loss_asp + 0.2 * var
    # loss = 0.7 * pol_loss + 0.2 * loss_asp + 0.1 * var
    # loss = 0.45 * pol_loss + 0.40 * loss_asp + 0.15 * var
    # loss = 0.5 * pol_loss + 0.4 * loss_asp + 0.10 * var
    loss = pol_loss + loss_asp + 0.1 * var
    # pdb.set_trace()
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()

    return loss.data.cpu().numpy()

