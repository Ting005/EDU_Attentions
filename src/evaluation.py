import pdb

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


def __calculate_matrix(lst_y_true, lst_y_pred):
    ma_f1 = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average='macro')
    f1s = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average=None, labels=[0, 1, 2])
    acc = accuracy_score(y_true=lst_y_true, y_pred=lst_y_pred)
    cfm = confusion_matrix(y_true=lst_y_true, y_pred=lst_y_pred, labels=[0, 1, 2])
    str_f1s = '{:.4f}|{:.4f}|{:.4f}'.format(f1s[0], f1s[1], f1s[2])
    return {'macro_f1': ma_f1, 'f1s': str_f1s, 'acc': acc, 'cfm': cfm}


def eval_edu_attention(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_seg, segs, num_word_seg, num_seg, batch_sent, num_word_sent, sent_order, batch_asplabels, num_asp, batch_pol_labels = inputs
            batch_seg, batch_sent, sent_order, num_asp = [v.to(_compute_device) for v in [batch_seg, batch_sent, sent_order, num_asp]]
            pol_logits, asp_prob, var, _ = _model(batch_seg, num_word_seg, num_seg, batch_sent, num_word_sent, sent_order)
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()
            y_asp_ture = batch_asplabels.numpy()
            # need to ignore na aspect
            # na_aspect_idx = y_asp_ture.shape[1] - 1
            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for asp_idx, (y_t, y_p, y_t_asp) in enumerate(zip(s_y_t, s_y_p, a_y_t)):
                    if y_t_asp > 0:# and asp_idx != na_aspect_idx:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_bert_edu_attention(_compute_device, _model, _eval_data, _asp_ids_and_mask):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            [seg_input_ids, seg_attention_mask], [batch_padded_sent_ids, batch_padded_sent_masks], num_segs, asp_labels, pol_labels = inputs
            asp_ids, asp_ids_mask = _asp_ids_and_mask
            seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels = \
                [v.to(_compute_device) for v in [seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels]]

            pol_logits, asp_prob, var = _model(num_segs, seg_input_ids, seg_attention_mask, batch_padded_sent_ids, batch_padded_sent_masks, asp_ids, asp_ids_mask)

            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)

            y_sem_true = pol_labels.cpu().numpy()
            y_asp_ture = asp_labels.cpu().numpy()
            # na_aspect_idx = y_asp_ture.shape[1] - 1

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                # for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                for asp_idx, (y_t, y_p, y_t_asp) in enumerate(zip(s_y_t, s_y_p, a_y_t)):
                    if y_t_asp > 0:# and asp_idx != na_aspect_idx:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result
