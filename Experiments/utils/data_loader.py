import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]

def read_cf_ali(file_name):
    return np.loadtxt(file_name, dtype=np.int32)

# def read_cf_gowalla(file_name):
#     inter_mat = list()
#     neighbor_dict = {}
#     deg_dict = {}
#     lines = open(file_name, 'r').readlines()
#     for l in lines:
#         tmps = l.strip()
#         inters = [int(i) for i in tmps.split(' ')]
#         u_id, pos_ids = inters[0], inters[1:]
#         neighbor_dict[u_id] = pos_ids
#         deg_dict[u_id] = len(pos_ids)
#         pos_ids = list(set(pos_ids))
#         for i_id in pos_ids:
#             inter_mat.append([u_id, i_id])
#     return np.array(inter_mat), neighbor_dict, deg_dict

def read_cf_gowalla(file_name):
    inter_mat = list()
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

def read_cf_gowalla_test(file_name):
    inter_mat = list()
    f = dict()
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
            if i_id not in f:
                f[i_id] = []
            f[i_id].append(u_id)

    # Get the number of items in the dictionary.
    num = list()
    for key, value in f.items():
        num.append(len(value))
    num.sort(reverse=True)
    perc_80 = num[int(len(num) * 0.2)]
    perc_5 = num[int(len(num) * 0.05)]

    # Create a list to store the item popularity labels.
    popular, normal, unpopular = [], [], []

    # Iterate over the items in the dictionary.
    for u_id, pos_ids in f.items():
        # Get the size of the item.
        num_pos_ids = len(pos_ids)

        # Label the item as unpopular, popular, or normal.
        if num_pos_ids < perc_80:
            unpopular.append(u_id)
        elif num_pos_ids > perc_5:
            popular.append(u_id)
        else:
            normal.append(u_id)
    print('pp', len(popular))
    print('nn', len(normal))
    print('un', len(unpopular))
    return np.array(inter_mat), popular, normal, unpopular

def read_cf_book(file_name):
    inter_mat = list()
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset != 'yelp2018' and dataset!='gowalla' and dataset!='amazon-book':
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

        


def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset == 'yelp2018':
        read_cf = read_cf_yelp2018
    elif dataset == 'gowalla':
        read_cf = read_cf_gowalla
        read_cf_test = read_cf_gowalla_test
    elif dataset == 'amazon':
        read_cf = read_cf_amazon
    elif dataset == 'amazon-book':
        read_cf = read_cf_book
    else:
        read_cf = read_cf_ali

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf, popular, normal, unpopular = read_cf_test(directory + 'test.txt')
    if args.dataset != 'yelp2018' and args.dataset!='gowalla' and args.dataset!='amazon-book':
        valid_cf = read_cf(directory + 'valid.txt')
    else:
        valid_cf = test_cf
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset != 'yelp2018' and args.dataset!='gowalla' and args.dataset!='amazon-book' else None,
        'test_user_set': test_user_set,
    }

    print('loading over ...')
    # return train_cf, user_dict, n_params, norm_mat, neighbor_dict, deg_dict
    return train_cf, user_dict, n_params, norm_mat
