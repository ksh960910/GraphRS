'''
Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F

# nohup python main.py --dataset gowalla --gnn sgcl --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --n_negs 1 --K 1 --tau 0.2 --lamb 0.5 --eps 0.1 &

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
        '''SimGCL'''
        ego_embeddings = torch.cat([user_embed, item_embed], dim=0)
        all_embeddings = []
        for k in range(self.n_hops):
            ego_embeddings = torch.sparse.mm(self.interact_mat, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings


class M1(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(M1, self).__init__()

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
        
        '''FN 부분'''
        self.fnk = args_config.fnk

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

        rec_user_emb, rec_item_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              perturbed = False,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user], rec_item_emb[pos_item], rec_item_emb[neg_item]
        s_user_view1, s_item_view1, s_user_view2, s_item_view2, s_user_view3, s_item_view3, s_user_view4, s_item_view4 = self.create_support_set()
        s1_user = self.get_user_sim_score(s_user_view1, rec_user_emb, user)
        s2_user = self.get_user_sim_score(s_user_view2, rec_user_emb, user)
        s3_user = self.get_user_sim_score(s_user_view3, rec_user_emb, user)
        s4_user = self.get_user_sim_score(s_user_view4, rec_user_emb, user)
        s1_item = self.get_item_sim_score(s_item_view1, rec_item_emb, pos_item)
        s2_item = self.get_item_sim_score(s_item_view2, rec_item_emb, pos_item)
        s3_item = self.get_item_sim_score(s_item_view3, rec_item_emb, pos_item)
        s4_item = self.get_item_sim_score(s_item_view4, rec_item_emb, pos_item)
        
        user_scores, _ = torch.stack([s1_user, s2_user, s3_user, s4_user]).max(dim=0)
        item_scores, _ = torch.stack([s1_item, s2_item, s3_item, s4_item]).max(dim=0)
        user_fn_idx = self.fn_detection(user_scores, self.fnk)
        item_fn_idx = self.fn_detection(item_scores, self.fnk)

        rec_loss =self.create_bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        batch_loss = rec_loss + self.l2_reg_loss(self.decay, user_emb, pos_item_emb)
        cl_loss = self.cl_rate * self.cal_cl_loss([user, pos_item], user_fn_idx, item_fn_idx)


        return batch_loss + cl_loss

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              perturbed=True,
                                              edge_dropout=False,
                                              mess_dropout=False)

        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))

        return torch.mean(loss)
    
    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg
    
    def cal_cl_loss(self, idx, user_fn_idx, item_fn_idx):
        u_idx = torch.unique(torch.LongTensor(idx[0].cpu()).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.LongTensor(idx[1].cpu()).type(torch.long)).to(self.device)
        user_view1, item_view1 = self.generate()
        user_view2, item_view2 = self.generate()
        # user_fn_idx : [batch_size, top-k]
        user_fn = user_view2[user_fn_idx,:]
        item_fn = item_view2[item_fn_idx,:]

        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], user_fn, self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], item_fn, self.temp)
        return user_cl_loss + item_cl_loss
    
    # def InfoNCE(self, view1, view2, fn, temperature, b_cos=True):
    #     if b_cos:
    #         view1, view2, fn = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(fn, dim=2)
    #     pos_score = (view1 * view2).sum(dim=-1)
    #     pos_score = torch.exp(pos_score / temperature)
    #     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    #     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    #     for i in range(fn.shape[1]):
    #         fn_view = fn[:,i,:] # [batch_size, emb_size]
    #         pos_score = (view1 * fn_view).sum(dim=-1)
    #         pos_score = torch.exp(pos_score / temperature)
    #         fn_cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    #         cl_loss += fn_cl_loss
    #     cl_loss/=(1+self.fnk)
            
    #     return torch.mean(cl_loss)

    def InfoNCE(self, view1, view2, fn, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    
    def create_support_set(self):
        s_user_view1, s_item_view1 = self.generate()  # [n_user,emb_size]
        s_user_view2, s_item_view2 = self.generate()
        s_user_view3, s_item_view3 = self.generate()
        s_user_view4, s_item_view4 = self.generate()
        return s_user_view1, s_item_view1, s_user_view2, s_item_view2, s_user_view3, s_item_view3, s_user_view4, s_item_view4
    
    def get_user_sim_score(self, support_user_view, user_gcn_emb, idx):
        u_idx = torch.unique(torch.LongTensor(idx.cpu()).type(torch.long)).to(self.device)
        # support view : [n_user, emb_size]
        # item_gcn_emb : [n_item, emb_size]
        score = torch.matmul(support_user_view[u_idx], user_gcn_emb[u_idx].t())
        return score

    def get_item_sim_score(self, support_item_view, item_gcn_emb, idx):
        i_idx = torch.unique(torch.LongTensor(idx.cpu()).type(torch.long)).to(self.device)
        score = torch.matmul(support_item_view[i_idx], item_gcn_emb[i_idx].t())
        return score
    
    def fn_detection(self, score, k):
        # FN이라고 판단한 node의 index를 return하는 함수
        '''top-k strategy'''
        sorted_score, indices = torch.sort(score, 1, descending=True)
        topk_mask = indices[:,1:k+1]
        '''threshold strategy'''
        return topk_mask
        

