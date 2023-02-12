import numpy as np
import pdb
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import torch.nn.functional as F


def __calculate_matrix(lst_y_true, lst_y_pred):

    ma_f1 = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average='macro')
    f1s = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average=None, labels=[0, 1, 2])
    acc = accuracy_score(y_true=lst_y_true, y_pred=lst_y_pred)
    cfm = confusion_matrix(y_true=lst_y_true, y_pred=lst_y_pred, labels=[0, 1, 2])
    str_f1s = '{:.2f}|{:.2f}|{:.2f}'.format(f1s[0] * 100, f1s[1] * 100, f1s[2] * 100)
    pdb.set_trace()
    # needs to calculte at each aspect level



    return {'macro_f1': ma_f1, 'f1s': str_f1s, 'acc': acc, 'cfm': cfm}


def eval_single_label_wo_asp(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data, start=1):

            batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels, batch_text = inputs
            pol_logits, scores = _model(batch_sent.to(_compute_device), num_sent_words.cpu())
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.cpu().numpy()

            lst_y_pred.extend(y_sem_pred.tolist())
            lst_y_true.extend(y_sem_true.tolist())

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_single_label(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data, start=1):

            batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels, batch_text = inputs

            pol_logits, scores = _model(batch_sent.to(_compute_device), num_sent_words, batch_asp_idxes.to(_compute_device))
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.cpu().numpy()

            lst_y_pred.extend(y_sem_pred.tolist())
            lst_y_true.extend(y_sem_true.tolist())

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_han_single_label(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data, start=1):
            batch_seg, segs, num_word_seg, num_seg, sent_order, batch_seg_asp, batch_pol_labels = inputs

            batch_seg, sent_order, batch_seg_asp = [v.to(_compute_device) for v in [batch_seg, sent_order, batch_seg_asp]]

            pol_logits, edu_att_scores, word_att_scores = _model(batch_seg, num_word_seg, num_seg, sent_order, batch_seg_asp)

            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()

            lst_y_pred.extend(y_sem_pred.tolist())
            lst_y_true.extend(y_sem_true.tolist())

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_single_label_bert(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data, start=1):
            batch_sent, sent_masks, batch_pol_labels = [v.to(_compute_device) for v in inputs]
            pol_logits = _model(batch_sent, sent_masks)

            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.cpu().numpy()
            lst_y_pred.extend(y_sem_pred.tolist())
            lst_y_true.extend(y_sem_true.tolist())

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_single_label_gac(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data, start=1):

            batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = inputs
            batch_sent, batch_asp_labels, batch_pol_labels = [v.to(_compute_device) for v in [batch_sent, batch_asp_labels, batch_pol_labels]]
            # pdb.set_trace()
            pol_logits, x, y = _model(batch_sent, batch_asp_labels)
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.cpu().numpy()

            lst_y_pred.extend(y_sem_pred.tolist())
            lst_y_true.extend(y_sem_true.tolist())

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_multilabel_han(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        total_loss = 0
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_seg, segs, num_word_seg, num_seg, sent_order, batch_asplabels, num_asp, batch_pol_labels = inputs
            batch_seg, sent_order, num_asp = [v.to(_compute_device) for v in [batch_seg, sent_order, num_asp]]
            # pdb.set_trace()
            pol_logits, asp_prob, var, _ = _model(batch_seg, num_word_seg, num_seg, sent_order, num_asp)
            # pol_loss = F.cross_entropy(pol_logits.permute(0, 2, 1), batch_pol_labels.cpu(), ignore_index=-1)
            # loss_asp = F.binary_cross_entropy(asp_prob, batch_asplabels)
            # loss = pol_loss + loss_asp + 0.15 * var  # performs better at mams
            # total_loss += loss * len(inputs)
            # pdb.set_trace()
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()
            y_asp_ture = batch_asplabels.numpy()

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                    if y_t_asp > 0:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result


def eval_multilabel_flat(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = inputs
            prob_asp, prob_sentiment, attn_asp, attn_sen = _model(batch_sent.to(_compute_device), num_sent_words)
            # prob_asp, prob_sentiment  = _model(batch_sent.to(_compute_device), num_sent_words)
            y_sem_pred = np.argmax(prob_sentiment.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()
            y_asp_ture = batch_asp_labels.numpy()

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                    if y_t_asp > 0:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result

def eval_multilabel_flat_can(_compute_device, _model, _eval_data):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_sent, num_sent_words, batch_asp_labels, batch_pol_labels = inputs
            prob_asp, prob_sentiment, _ = _model(batch_sent.to(_compute_device), num_sent_words)
            # prob_asp, prob_sentiment  = _model(batch_sent.to(_compute_device), num_sent_words)
            y_sem_pred = np.argmax(prob_sentiment.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()
            y_asp_ture = batch_asp_labels.numpy()

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                    if y_t_asp > 0:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result

def eval_multilabel_han_bert(_compute_device, _model, _eval_data, _asp_ids_and_mask):
    _model.eval()
    with torch.no_grad():
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            [seg_input_ids, seg_attention_mask], num_segs, asp_labels, pol_labels = inputs
            asp_ids, asp_ids_mask = _asp_ids_and_mask
            seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels = \
                [v.to(_compute_device) for v in [seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask, num_segs, asp_labels, pol_labels]]

            pol_logits, asp_prob, var = _model(num_segs, seg_input_ids, seg_attention_mask, asp_ids, asp_ids_mask)

            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)

            y_sem_true = pol_labels.cpu().numpy()
            y_asp_ture = asp_labels.cpu().numpy()

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                    if y_t_asp > 0:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        result = __calculate_matrix(lst_y_true=lst_y_true, lst_y_pred=lst_y_pred)
        return result
