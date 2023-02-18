import numpy as np
import scipy.sparse as sp
from utils.node_copying import generate_graph

from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)

def generate_sampled_graph(directory):
    try:
        zeta = np.load(directory + 'zeta/zeta.npy')
        sampled_graph = directory + 'sampled_graph/sampled_graph_epoch' + str(1) + '_epsilon' + str(0.01)
        f = open(sampled_graph, 'r')
        f.close()
        # generate_graph(self.path).generate_graph(zeta=zeta, epsilon=0.01, iteration=iteration)
        # sampled_graph = self.path + '/sampled_graph/sampled_graph_' + str(iteration+1)
    
    except Exception:
        zeta = generate_graph(directory).node_copying()
        generate_graph(directory).generate_graph(zeta=zeta, epsilon=0.01, iteration=1)
        # sampled_graph = self.path + '/sampled_graph/sampled_graph_' + str(iteration+1)
    
    sampled_graph = directory + 'sampled_graph/sampled_graph_epoch' + str(1) + '_epsilon' + str(0.01)

    return sampled_graph

def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]

def read_cf_movielens(file_name):
    return np.loadtxt(file_name, dtype=np.int32)

def read_cf_gowalla_train(directory):
    sampled_graph = generate_sampled_graph(directory)
    print(sampled_graph)

    inter_mat = list()
    all_pos_ids = list()
    lines = open(sampled_graph, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
            all_pos_ids.append(i_id)
            # if i_id not in all_pos_ids:
            #     all_pos_ids.append(i_id)
    return np.array(inter_mat), list(set(all_pos_ids))


def read_cf_gowalla(file_name):
    inter_mat = list()
    all_pos_ids = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
            all_pos_ids.append(i_id)
    return np.array(inter_mat), list(set(all_pos_ids))


def read_cf_yelp2018_train(directory):
    sampled_graph = generate_sampled_graph(directory)
    print(sampled_graph)

    inter_mat = list()
    lines = open(sampled_graph, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

def read_cf_yelp2018(file_name):
    inter_mat = list()
    all_pos_ids = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
            all_pos_ids.append(i_id)
    return np.array(inter_mat), list(set(all_pos_ids))


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset != 'yelp2018' and dataset != 'movielens' and dataset !='gowalla':
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

def count_item_degree(all_pos_ids, inter_mat):
    deg_item = {}
    all_pos_ids.sort()
    for i in all_pos_ids:
        deg_item[i] = 0
    for tup in inter_mat:
        deg_item[tup[1]]+=1
    return deg_item


    # inter_mat = list()
    # lines = open(sampled_graph, "r").readlines()
    # count_item_deg = {}
    # for l in lines:
    #     tmps = l.strip()
    #     inters = [int(i) for i in tmps.split(" ")] # 0 12 15 28 589 ---
    #     u_id, pos_ids = inters[0], inters[1:]
    #     pos_ids = list(set(pos_ids))
    #     for i_id in pos_ids:
    #         count_item_deg[i_id]
            # inter_mat.append([u_id, i_id])
        
    # return np.array(inter_mat)

def load_data(model_args, sample_method=0):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if sample_method==1:
        if dataset == 'yelp2018':
            read_cf = read_cf_yelp2018_train
            read_test_cf = read_cf_yelp2018
        elif dataset == 'gowalla':
            read_cf = read_cf_gowalla_train
            read_test_cf = read_cf_gowalla
    else:
        if dataset == 'yelp2018':
            read_cf = read_cf_yelp2018
        elif dataset == 'movielens':
            read_cf = read_cf_movielens
        elif dataset == 'gowalla':
            read_cf = read_cf_gowalla
        elif dataset == 'amazon':
            read_cf = read_cf_amazon

    print('reading train and test user-item set ...')
    if sample_method==1:
        train_cf, all_pos_ids = read_cf(directory)
        deg_item = count_item_degree(all_pos_ids, train_cf)
        test_cf, all_pos_ids = read_test_cf(directory + 'test.txt')
    else:
        train_cf, all_pos_ids = read_cf(directory + 'train.txt')
        deg_item = count_item_degree(all_pos_ids, train_cf)
        test_cf, all_pos_ids = read_cf(directory + 'test.txt')
        
    if args.dataset!='gowalla' and args.dataset != 'yelp2018' and args.dataset != 'movielens':
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
        'valid_user_set': valid_user_set if args.dataset != 'gowalla' and args.dataset != 'yelp2018' and args.dataset != 'movielens' else None,
        'test_user_set': test_user_set,
    }

    print('loading over ...')
    return train_cf, user_dict, n_params, norm_mat, deg_item

