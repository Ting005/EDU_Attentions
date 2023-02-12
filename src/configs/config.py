import argparse, logging
import numpy as np
import random
import torch
from tensorboard_logger import configure


parser = argparse.ArgumentParser()
parser.add_argument('--voc_size', type=int, default=32768)
parser.add_argument('--dim_word', type=int, default=300, choices=[300])
parser.add_argument('--dim_hidden', type=int, default=256, choices=[256])
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_label', type=int, default=3, choices=[3])
parser.add_argument('--n_aspect', type=int, default=22, choices=[5])
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--cell_dropout', type=float, default=0.5)
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--max_len_sen', type=int, default=4)
# parser.add_argument('--max_len_edu', type=int, default=20)

parser.add_argument('--iter_num', type=int, default=8*320)
parser.add_argument('--per_checkpoint', type=int, default=16)
# parser.add_argument('--seed', type=int, default=2018)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--data', type=str, default='rest')
parser.add_argument('--const_random_sate', type=int, default=42)
# parser.add_argument('--path_wordvec', type=str, default='vectors.glove.840B.txt')
parser.add_argument('--name_model', type=str, default='run_han')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--path_wordvec', type=str, default='vectors.glove.840B.txt')

FLAGS = parser.parse_args()
configure("runs/summary/%s" % FLAGS.name_model, flush_secs=3)
logging.basicConfig(filename='runs/log/%s.log' % FLAGS.name_model, level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.info('model parameters: {}'.format(FLAGS))

print('seed everything')
torch.manual_seed(FLAGS.const_random_sate)
random.seed(FLAGS.const_random_sate)
np.random.seed(seed=FLAGS.const_random_sate)
torch.manual_seed(FLAGS.const_random_sate)
torch.cuda.manual_seed(FLAGS.const_random_sate)
torch.cuda.manual_seed_all(FLAGS.const_random_sate)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

compute_device = torch.device("cuda:{}".format(FLAGS.gpu) if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()

print('compute device:', compute_device)
print(FLAGS)
