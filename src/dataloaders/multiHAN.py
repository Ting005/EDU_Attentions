import pdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random


class MultiHAN(Dataset):

    def __init__(self, _input_data, asp_dict):
        self.dataset = _input_data
        self.n_asp = len(asp_dict)
        self.n_pol = 3
        self.asp_dict = asp_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]
        segs = instance['segs']
        segs2idx = instance['segs2idx']
        num_seg = len(segs2idx)
        num_asp = len(instance['asp_pol_idx'])
        len_seg_words = [len(edu) for edu in segs2idx]
        text2idx = [w for seg in segs2idx for w in seg]
        text = [w for seg in segs for w in seg.split()]
        num_sent_words = len(text2idx)
        # pdb.set_trace()
        asp2idx = np.zeros(len(self.asp_dict), dtype='int64')
        asp_labels = np.zeros(len(self.asp_dict), dtype='int64')
        pol_labels = np.ones(len(self.asp_dict), dtype='int64') * (-1) # change to 0 for experiment
        for asppol in instance['asp_pol_idx']:
            asp = asppol['asp']
            pol2idx = asppol['pol2idx']
            # pdb.set_trace()
            asp_labels[self.asp_dict[asp]] = 1
            pol_labels[self.asp_dict[asp]] = pol2idx
            asp2idx[self.asp_dict[asp]] = asppol['asp2idx']
            # pol_labels.append(pol2idx)
        assert sum([1 if p == 1 else 0 for p in asp_labels]) == num_asp
        # pdb.set_trace()
        return segs2idx, text2idx, text, segs, num_seg, num_asp, num_sent_words, len_seg_words, asp_labels, pol_labels


class DatasetLoader(object):
    def _tuple_batch_edu(self, batch_data):
        sorted_batch = sorted(batch_data, key=lambda b: b[4], reverse=True)
        # for batch edu, sorted by number of words in each edu
        segs2idx, _, _, segs, num_seg, num_asp, _, _, asp_labels, pol_labels = zip(*sorted_batch)
        # pdb.set_trace()
        # sorted by number words in edu, with sent2idx which sorted by sentence length
        seg_stat = sorted([(len(seg), s_idx, seg_idx, seg) for s_idx, lst_segs in enumerate(segs2idx)
                           for seg_idx, seg in enumerate(lst_segs)], reverse=True)
        # padding
        max_num_seg, max_num_word = num_seg[0], seg_stat[0][0]
        batch_seg = torch.zeros(len(seg_stat), max_num_word).long()
        sent_order = torch.zeros(len(segs2idx), max_num_seg).long()
        num_word_seg = [s[0] for s in seg_stat]
        # pdb.set_trace()
        for seg_stat_idx, (len_seg, s_idx, seg_idx, seg) in enumerate(seg_stat):
            batch_seg[seg_stat_idx, 0: len_seg] = torch.Tensor(seg).long()
            sent_order[s_idx, seg_idx] = seg_stat_idx + 1  # 0 is for padding

        num_asp = torch.LongTensor(num_asp)
        num_seg = torch.LongTensor(num_seg)
        num_word_seg = torch.Tensor(num_word_seg).long()
        batch_pol_labels = torch.tensor(pol_labels)
        batch_asplabels = torch.tensor(asp_labels).float()
        # pdb.set_trace()

        return batch_seg, segs, num_word_seg, num_seg, sent_order, batch_asplabels, num_asp, batch_pol_labels

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        print('seed worker')

    def __init__(self, batch_size, asp_dict, input_data):
        g = torch.Generator()
        g.manual_seed(42)
        _dataset = MultiHAN(input_data, asp_dict)
        self.num_batch = len(_dataset) // batch_size
        print(self.num_batch)
        self.Data = DataLoader(_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=self._tuple_batch_edu,
                               pin_memory=True, worker_init_fn=self.seed_worker, generator=g)
