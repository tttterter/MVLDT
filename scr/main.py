import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train
import torch.backends.cudnn as cudnn


# #  music
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='book',help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--n_layer', type=int, default=2, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')

parser.add_argument('--dim', type=int, default=256, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=32, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=64, help='the number of triples in triple set of item')
parser.add_argument('--num_users', type=int, default=20000000000, help='the number of user')
parser.add_argument('--num_items', type=int, default=10000000000, help='the number of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')
#

#book
# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset', type=str, default='book',help='which dataset to use (music, book, movie, restaurant)')
# parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
# parser.add_argument('--n_layer', type=int, default=2, help='depth of layer')
# parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
#
# parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
# parser.add_argument('--user_triple_set_size', type=int, default=16, help='the number of triples in triple set of user')
# parser.add_argument('--item_triple_set_size', type=int, default=32, help='the number of triples in triple set of item')
# parser.add_argument('--num_users', type=int, default=20000000000, help='the number of user')
# parser.add_argument('--num_items', type=int, default=10000000000, help='the number of item')
# parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')
#
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')



# movie
# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset', type=str, default='movie1m', help='which dataset to use (music, book, movie, restaurant)')
# parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
# parser.add_argument('--n_layer', type=int, default=2, help='depth of layer')
# parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
#
# parser.add_argument('--dim', type=int, default=6464 , help='dimension of entity and relation embeddings')
# parser.add_argument('--user_triple_set_size', type=int, default=128, help='the number of triples in triple set of user')
# parser.add_argument('--item_triple_set_size', type=int, default=128, help='the number of triples in triple set of item')
# parser.add_argument('--num_users', type=int, default=20000000000, help='the number of user')
# parser.add_argument('--num_items', type=int, default=10000000000, help='the number of item')
# parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')
#
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)


    #
    # cudnn.deterministic = True
    # cudnn.benchmark = False


if not args.random_flag:
    set_random_seed(304, 2019)

data_info = load_data(args)
train(args, data_info)
