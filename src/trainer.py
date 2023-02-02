import torch.nn.functional as F


def train_edu_attention(_compute_device, _model, _optimizer, _inputs, _total_num_asp):
    _model.train()
    # batch_seg, segs, num_word_seg, num_seg, num_word_sent, sent_order, batch_asplabels, num_asp, batch_pol_labels
    batch_seg, segs, num_word_seg, num_seg, batch_sent, num_word_sent, sent_order, batch_asplabels, num_asp, batch_pol_labels = _inputs
    batch_seg, batch_sent, sent_order, num_asp, batch_pol_labels, batch_asplabels = [v.to(_compute_device) for v in [batch_seg, batch_sent, sent_order, num_asp, batch_pol_labels, batch_asplabels]]
    pol_logits, asp_prob, var, _ = _model(batch_seg, num_word_seg, num_seg, batch_sent, num_word_sent, sent_order)
    pol_loss = F.cross_entropy(pol_logits.permute(0, 2, 1), batch_pol_labels, ignore_index=-1)
    loss_asp = F.binary_cross_entropy(asp_prob, batch_asplabels)
    loss = 0.6 * pol_loss + 0.3 * loss_asp + 0.9 * var
    # loss = pol_loss + loss_asp + var
    # loss = pol_loss + loss_asp + 0.2 * var
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()

    return loss.data.cpu().numpy()


def train_bert_edu_attention(_compute_device, _model, _optimizer, _inputs, _total_num_asp, asp_ids_and_mask=None):
    _model.train()
    [seg_input_ids, seg_attention_mask], [batch_padded_sent_ids, batch_padded_sent_masks], num_segs, asp_labels, pol_labels = _inputs
    asp_ids, asp_ids_mask = asp_ids_and_mask
    seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels = \
        [v.to(_compute_device) for v in [seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels]]
    pol_logits, asp_prob, var = _model(num_segs, seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask)
    loss_asp = F.binary_cross_entropy(asp_prob, asp_labels)
    pol_loss = F.cross_entropy(pol_logits.permute(0, 2, 1), pol_labels, ignore_index=-1)
    # loss = 0.5 * pol_loss + 0.4 * loss_asp + 0.10 * var
    # loss = pol_loss + loss_asp + 0.10 * var
    loss = pol_loss + loss_asp + 0.1 * var
    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()

    return loss.data.cpu().numpy()
