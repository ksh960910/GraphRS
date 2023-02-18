'''
Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)

python main.py --dataset gowalla --gnn gcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --K 1 --tau 0.2 --lamb 0.5 --eps 0.1
'''
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, n_items, interact_mat, eps, layer_cl,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        '''추가'''
        self.eps = eps
        self.layer_cl = layer_cl

        self.n_users = n_users
        self.n_items = n_items
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed, perturbed=False,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        # all_embed = torch.cat([user_embed, item_embed], dim=0)
        # agg_embed = all_embed
        '''0-th layer 추가'''
        # embs = [all_embed]

        # for hop in range(self.n_hops):
        #     interact_mat = self._sparse_dropout(self.interact_mat,
        #                                         self.edge_dropout_rate) if edge_dropout \
        #                                                                 else self.interact_mat

        #     agg_embed = torch.sparse.mm(interact_mat, agg_embed)
        #     if mess_dropout:
        #         agg_embed = self.dropout(agg_embed)
        #     # agg_embed = F.normalize(agg_embed)
        #     embs.append(agg_embed)
        # embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        # return embs[:self.n_users, :], embs[self.n_users:, :]

        ego_embeddings = torch.cat([user_embed, item_embed], dim=0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_hops):
            ego_embeddings = torch.sparse.mm(self.interact_mat, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)

            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        user_gcn_emb, item_gcn_emb = final_embeddings[:self.n_users, :], final_embeddings[self.n_users:, :]
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.n_users, self.n_items])
        if perturbed:
            return user_gcn_emb, item_gcn_emb, user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings


class MiXSimGCL(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(MiXSimGCL, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K

        '''추가한 부분'''
        self.temp = args_config.tau
        self.eps = args_config.eps
        self.cl_rate = args_config.lamb
        self.layer_cl = args_config.cll

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         interact_mat=self.sparse_norm_adj,
                         eps=self.eps,
                         layer_cl=self.layer_cl,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb, rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = self.gcn(self.user_embed,
                                                                                                    self.item_embed,
                                                                                                    perturbed = True,
                                                                                                    edge_dropout=self.edge_dropout,
                                                                                                    mess_dropout=self.mess_dropout)

        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user], rec_item_emb[pos_item], rec_item_emb[neg_item]

        
        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.negative_sampling(user_gcn_emb, item_gcn_emb,
                                                           user, neg_item[:, k*self.n_negs: (k+1)*self.n_negs],
                                                           pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        bpr_loss = self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)
        cl_loss = self.cl_rate * self.cal_cl_loss([user, pos_item],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)

        return bpr_loss + cl_loss

        # rec_loss =self.create_bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        # batch_loss = rec_loss + self.l2_reg_loss(self.decay, user_emb, pos_item_emb)
        # cl_loss = self.cl_rate * self.cal_cl_loss([user, pos_item],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)

        # return batch_loss + cl_loss

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              perturbed=False,
                                              edge_dropout=False,
                                              mess_dropout=False)
        # user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        # return mf_loss + emb_loss, mf_loss, emb_loss
        return mf_loss + emb_loss

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        # u_idx = torch.unique(torch.LongTensor(idx[0].cpu()).type(torch.long)).to(self.device)
        # i_idx = torch.unique(torch.LongTensor(idx[1].cpu()).type(torch.long)).to(self.device)
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)

