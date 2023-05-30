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


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, deg_item = load_data(args, sample_method=0)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    train_cf_1, user_dict_1, n_params_1, norm_mat_1, deg_item_1 = load_data(args, args.sample_method, epoch=1)
    train_cf_size_1 = len(train_cf_1)
    train_cf_1 = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_1], np.int32))

    train_cf_2, user_dict_2, n_params, norm_mat_2, deg_item_2 = load_data(args, args.sample_method, epoch=2)
    train_cf_size = len(train_cf_2)
    train_cf_2 = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_2], np.int32))

    # train_cf_3, user_dict_3, n_params, norm_mat_3, deg_item_3 = load_data(args, args.sample_method, epoch=3)
    # train_cf_size = len(train_cf_3)
    # train_cf_3 = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_3], np.int32))
    

    '''embconcat data'''
    # train_cf_obs, user_dict_obs, n_params_obs, norm_mat_obs, deg_item_obs = load_data(args,sample_method=0)
    # train_cf_size_obs = len(train_cf_obs)
    # train_cf_obs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf_obs], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.LightGCN import LightGCN, TestGCN
    from modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        # model = LightGCN(n_params, args, norm_mat, norm_mat_1, norm_mat_2, norm_mat_3).to(device)
        model = LightGCN(n_params, args, norm_mat, norm_mat_1, norm_mat_2).to(device)
        '''embconcat layer'''
    # elif args.gnn == 'testgcn':
    #     model = TestGCN(n_params, args, norm_mat, norm_mat_obs).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        # if epoch%4==0:
        #     user_dict = user_dict
        #     deg_item = deg_item
        #     train_cf_ = train_cf
        #     index = np.arange(len(train_cf_))
        #     np.random.shuffle(index)
        #     train_cf_ = train_cf_[index].to(device)
        # elif epoch%4==1:
        #     user_dict = user_dict_1
        #     # deg_item = deg_item_1
        #     train_cf_ = train_cf_1
        #     index = np.arange(len(train_cf_))
        #     np.random.shuffle(index)
        #     train_cf_ = train_cf_[index].to(device)
        # elif epoch%4==2:
        #     user_dict = user_dict_2
        #     # deg_item = deg_item_2
        #     train_cf_ = train_cf_2
        #     index = np.arange(len(train_cf_))
        #     np.random.shuffle(index)
        #     train_cf_ = train_cf_[index].to(device)
        # elif epoch%4==3:
        #     user_dict = user_dict_3
        #     # deg_item = deg_item_3
        #     train_cf_ = train_cf_3
        #     index = np.arange(len(train_cf_))
        #     np.random.shuffle(index)
        #     train_cf_ = train_cf_[index].to(device)
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)
        
        train_cf_ = train_cf_1
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_1 = train_cf_[index].to(device)

        train_cf_ = train_cf_2
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_2 = train_cf_[index].to(device)

        # train_cf_ = train_cf_3
        # index = np.arange(len(train_cf_))
        # np.random.shuffle(index)
        # train_cf_3 = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        '''embconcat'''
        # while s + args.batch_size <= len(train_cf_obs):
        while s + args.batch_size <= len(train_cf_):
            '''Compute mean of the embeddings'''
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)
            batch_1 = get_feed_dict(train_cf_1,
                                  user_dict_1['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)
            batch_2 = get_feed_dict(train_cf_2,
                                  user_dict_2['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)
            # batch_3 = get_feed_dict(train_cf_3,
            #                       user_dict_3['train_user_set'],
            #                       s, s + args.batch_size,
            #                       n_negs)                

            '''embconcat'''
            # batch_obs = get_feed_dict(train_cf_obs,
            #                           user_dict_obs['train_user_set'],
            #                           s, s + args.batch_size,
            #                           n_negs)

            # batch_loss, _, _ = model(batch, batch_obs)
            batch_loss, _, _ = model(batch=batch, batch_1=batch_1, batch_2=batch_2)
            # batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", 'novelty', "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, deg_item, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['novelty'], test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, deg_item, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['novelty'], valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))