import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n):
    '''XSimGCL'''
    def sampling(user_item, train_set, n=1):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                negitem = random.choice(range(n_items))
                while negitem in train_set[user]:
                    negitem = random.choice(range(n_items))
                neg_items.append(negitem)
        return neg_items
    
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set)).to(device)
    return feed_dict

'''MGCL negative candidate selection'''
def user_negative_sampling(n):
    neg_candidate = []
    for i in range(n_users):
        negitems = []
        for j in range(n):
            while True:
                negitem = random.choice(range(n_users))
                if negitem != i:
                    break
            negitems.append(negitem)
        neg_candidate.append(negitems)
        
    return neg_candidate

def item_negative_sampling(n):
    neg_candidate = []
    for i in range(n_items):
        negitems = []
        for j in range(n):
            while True:
                negitem = random.choice(range(n_items))
                if negitem != i:
                    break
            negitems.append(negitem)
        neg_candidate.append(negitems)

    return neg_candidate


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''empty cache'''
    torch.cuda.empty_cache()

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.M1 import M1
    from modules.XSimGCL import XSimGCL
    from modules.SimGCL import SimGCL
    from modules.LightGCN import LightGCN
    from modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        print('MixGCF-LightGCN model setup')
        model = LightGCN(n_params, args, norm_mat).to(device)
    elif args.gnn == 'sgcl':
        print('Simple GCL model setup')
        model = SimGCL(n_params, args, norm_mat).to(device)
    elif args.gnn == 'xgcl':
        print('XSimple GCL model setup')
        model = XSimGCL(n_params, args, norm_mat).to(device)
    elif args.gnn == 'm1':
        print('Method 1 GCL model setup')
        model = M1(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    t = time()

    print("start training ...")

    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        '''mixup시에 필요한 negative candidate selection'''
        # neg_candidate_users = user_negative_sampling(n_negs)
        # neg_candidate_items = item_negative_sampling(n_negs)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            '''other model setup'''
            batch_loss = model(batch)

            '''xsimgcl model setup'''
            # batch_loss = model(neg_candidate_users, neg_candidate_items, batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            s += args.batch_size
        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))