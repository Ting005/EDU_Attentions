# https://likegeeks.com/seaborn-heatmap-tutorial/
import pdb

import numpy as np; np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sb; sb.set_theme()
import pickle


def heap_vis(_idx, _sentence, _lst_asp_inst):
    def __format_score(_text, _score):
        out_str = ''
        if _text != '<pad>':
            _str_score = '{:.2f}'.format(_score * 100) if _score > 0 else '0'
            out_str = "{0}\n{1}".format(_text, _str_score)
            # out_str = "{0}{1}".format(_text, '(0)' if _score == 0 else '')
            # out_str = _text

        return out_str

    color = sb.color_palette("Blues", 25)
    print(len(_lst_asp_inst))
    plt.figure(figsize=(500, 300))
    fig, axs = plt.subplots(len(_lst_asp_inst))
    # heap map with multiple sub-figures
    for f_idx, asp_inst in enumerate(_lst_asp_inst):
        segs = asp_inst['text']
        word_scores = asp_inst['word_scores']
        edu_scores = asp_inst['edu_scores']

        for edu_idx in range(0, edu_scores.shape[0]):
            edu_s = edu_scores[edu_idx]
            for w_idx in range(0,  word_scores.shape[1]):
                # if idx == 22:
                #     pdb.set_trace()
                word_scores[edu_idx, w_idx] = word_scores[edu_idx, w_idx] * edu_s

        asp_label = asp_inst['asp_label']
        num_inst, num_words = segs.shape
        # pdb.set_trace()
        labels = (np.asarray([__format_score(text, data) for text, data in zip(segs.flatten(), word_scores.flatten())])).reshape(num_inst, num_words)
        heat_map = sb.heatmap(word_scores, annot=labels, fmt='',   cbar=False, yticklabels=False,
                              xticklabels=[asp_label], annot_kws={'size':9}, linewidths=.5, cmap=color, ax=axs[f_idx])
        # axs[f_idx].set_title(asp_label, loc='bottom')
        # heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45, fontsize=11)

    # # plt.figure(figsize=(20, 5))
    # # sb.color_palette("light:b", as_cmap=True)
    # num_inst, num_words = scores.shape
    # labels = (np.asarray(["{0}\n{1}".format(text, __format_score(data)) for text, data in zip(texts.flatten(), scores.flatten())])).reshape(num_inst, num_words)
    # heat_map = sb.heatmap(scores, annot=labels, fmt='', cbar=False, yticklabels=aspects,
    #                       xticklabels=texts[0], annot_kws={'size': 12}, cmap="YlGnBu")
    # heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45, fontsize=11)
    plt.tight_layout()
    # plt.show()
    return heat_map


if __name__ == '__main__':
    rest_han = pickle.load(open('../model_files/rest_han_reg.raw', 'rb'))
    hard = rest_han['hard']

    lst_hard_instances = {}
    for idx, inst in enumerate(hard):
        text = inst['text']
        word_score = inst['word_score']
        edu_score = inst['edu_score']
        total_seg = inst['total_seg']
        labels = inst['labels']
        asp_true = inst['asp_true']
        sem_true = inst['sem_true']
        asp_counter = 0
        lst_asp_inst = []
        for asp_idx, asp_ind in enumerate(asp_true):
            if asp_ind == 1.0:
                asp_word_score = word_score[asp_idx, :]
                # max_num_segs, max_num_words = asp_word_score.shape
                segs = [seg.split() for seg in text]
                max_num_words = max(len(ss) for ss in segs)
                segs_padded = [ss + ['<pad>']*(max_num_words - len(ss)) for ss in segs]
                # pdb.set_trace()
                segs_padded = np.asarray(segs_padded)
                asp_label = labels[asp_counter]
                asp_counter += 1
                word_scores = asp_word_score[0:len(segs), 0:max_num_words]
                asp_edu_score = edu_score[asp_idx, :][0: len(segs)]

                asp_inst = {'text': segs_padded, 'word_scores': word_scores, 'edu_scores': asp_edu_score, 'asp_label': asp_label}
                lst_asp_inst.append(asp_inst)

                # texts = np.asarray([inst['text'] for inst in items])
                # asps_t_p = zip([inst['aspect'] for inst in items], np.asarray([inst['sem_true'] for inst in items]), np.asarray([inst['sem_pred'] for inst in items]))
                # asps = np.asarray(['{} ({} / {})'.format(a, polarity_dict[t], polarity_dict[p]) for (a, t, p) in asps_t_p])
                # asps = np.asarray([inst['aspect'] for inst in items])
                # sem_pred = np.asarray([inst['sem_pred'] for inst in items])
                # sem_true = np.asarray([inst['sem_true'] for inst in items])
                # scores = np.asarray([inst['scores'][0: len(texts[0])] for inst in items], dtype=float)
        pltobj = heap_vis(idx, ''.join(text), lst_asp_inst)
        pltobj.figure.savefig('./images/han_reg/rest_{}.jpeg'.format(idx))
        # pdb.set_trace()

    # polarity_dict = {0: 'neg', 1: 'neu', 2: 'neg'}
    # # convert to consumable format
    # for idx, (key, items) in enumerate(lst_hard_instances.items()):
    #     texts = np.asarray([inst['text'] for inst in items])
    #
    #     asps_t_p = zip([inst['aspect'] for inst in items], np.asarray([inst['sem_true'] for inst in items]), np.asarray([inst['sem_pred'] for inst in items]))
    #     asps = np.asarray(['{} ({} / {})'.format(a, polarity_dict[t], polarity_dict[p]) for (a, t, p) in asps_t_p])
    #
    #     scores = np.asarray([inst['scores'][0: len(texts[0])] for inst in items], dtype=float)
    #     pltobj = heap_vis(texts, scores, asps)
    #     # pltobj.figure.savefig('./images/atae_rest_{}.pdf'.format(idx))
    #     pltobj.figure.savefig('./images/atae/rest_{}.jpeg'.format(idx))
    #     # pdb.set_trace()

    print('finished')






