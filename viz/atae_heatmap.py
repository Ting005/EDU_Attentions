# https://likegeeks.com/seaborn-heatmap-tutorial/
import pdb

import numpy as np; np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sb; sb.set_theme()
import pickle


def heap_vis(texts, scores, aspects):
    def __format_score(_score):
        out_str = '0'
        if _score > 0:
            out_str = '{:.2f}'.format(_score * 100)
        return out_str
    # text = np.asarray([['a', 'b', 'c', 'd', 'e', 'f'], ['g', 'h', 'i', 'j', 'k', 'l'], ['m', 'n', 'o', 'p', 'q', 'r'], ['s', 't', 'u', 'v', 'w', 'x']])
    # data = np.random.rand(4, 6)
    # labels = (np.asarray(["{0}\n{1:.2f}".format(text, data) for text, data in zip(text.flatten(), data.flatten())])).reshape(4, 6)
    # heat_map = sb.heatmap(data, annot=labels, fmt='', cbar=False, yticklabels=['a', 'b', 'c', 'd'], xticklabels=False)
    # plt.show()
    # sb.set(font_scale=0.5)
    # fig, ax = plt.subplots(figsize=(50, 10))
    plt.figure(figsize=(20, 5))
    sb.color_palette("light:b", as_cmap=True)
    num_inst, num_words = scores.shape
    labels = (np.asarray(["{0}\n{1}".format(text, __format_score(data)) for text, data in zip(texts.flatten(), scores.flatten())])).reshape(num_inst, num_words)
    heat_map = sb.heatmap(scores, annot=labels, fmt='', cbar=False, yticklabels=aspects,
                          xticklabels=texts[0], annot_kws={'size': 12}, cmap="YlGnBu")
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    # plt.show()

    return heat_map


if __name__ == '__main__':
    rest_han = pickle.load(open('../model_files/rest_atae.raw', 'rb'))
    hard = rest_han['hard']

    lst_hard_instances = {}
    for inst in hard:
        text = inst['text']
        aspect = inst['aspect']
        scores = inst['scores']
        sem_pred = inst['sem_pred']
        sem_true = inst['sem_true']

        text_key = ' '.join(text)
        if lst_hard_instances.__contains__(text_key):
            lst_hard_instances[text_key].append(inst)
        else:
            lst_hard_instances[text_key] = [inst]
    polarity_dict = {0: 'neg', 1: 'neu', 2: 'pos'}
    # convert to consumable format
    for idx, (key, items) in enumerate(lst_hard_instances.items()):
        texts = np.asarray([inst['text'] for inst in items])

        asps_t_p = zip([inst['aspect'] for inst in items], np.asarray([inst['sem_true'] for inst in items]), np.asarray([inst['sem_pred'] for inst in items]))
        asps = np.asarray(['{} ({} / {})'.format(a, polarity_dict[t], polarity_dict[p]) for (a, t, p) in asps_t_p])

        # asps = np.asarray([inst['aspect'] for inst in items])
        # sem_pred = np.asarray([inst['sem_pred'] for inst in items])
        # sem_true = np.asarray([inst['sem_true'] for inst in items])
        scores = np.asarray([inst['scores'][0: len(texts[0])] for inst in items], dtype=float)
        pltobj = heap_vis(texts, scores, asps)
        # pltobj.figure.savefig('./images/atae_rest_{}.pdf'.format(idx))
        pltobj.figure.savefig('./images/atae/rest_{}.jpeg'.format(idx))
        # pdb.set_trace()

    print('finished')






