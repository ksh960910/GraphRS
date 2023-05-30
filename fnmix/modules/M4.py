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
        layer_embeddings = ego_embeddings
        for k in range(self.n_hops):
            ego_embeddings = torch.sparse.mm(self.interact_mat, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==0:
                layer_embeddings = ego_embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        user_layer_embeddings, item_layer_embeddings = torch.split(layer_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, user_layer_embeddings, item_layer_embeddings


class M4(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(M4, self).__init__()

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

    def forward(self, neg_candidate_users, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]

        rec_user_emb, rec_item_emb, layer_user_emb, _ = self.gcn(self.user_embed,
                                                                              self.item_embed,
                                                                              perturbed = False,
                                                                              edge_dropout=self.edge_dropout,
                                                                              mess_dropout=self.mess_dropout)
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user], rec_item_emb[pos_item], rec_item_emb[neg_item]

        rec_loss =self.create_bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        batch_loss = rec_loss + self.l2_reg_loss(self.decay, user_emb, pos_item_emb)
        cl_loss = self.cl_rate * self.cal_cl_loss([user, pos_item], layer_user_emb, neg_candidate_users)


        return batch_loss + cl_loss

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb, layer_user_emb, _ = self.gcn(self.user_embed,
                                                                 self.item_embed,
                                                                 perturbed=True,
                                                                 edge_dropout=False,
                                                                 mess_dropout=False)

        if split:
            return user_gcn_emb, item_gcn_emb, layer_user_emb
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
    
    def users_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_users):
        '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
           neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
        
        # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
        neg_emb = neg_gcn_emb[neg_candidate_users] # [n_users, n_negs, emb_size]

        '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
        alpha = (torch.rand_like(neg_emb)/2).cuda()
        # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
        neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
        scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach()
        neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]

        return neg_emb
    
    def cal_cl_loss(self, idx, layer_user_emb, neg_candidate_users):
        u_idx = torch.unique(torch.LongTensor(idx[0].cpu()).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.LongTensor(idx[1].cpu()).type(torch.long)).to(self.device)
        user_view1, item_view1, layer_user_view1 = self.generate()
        user_view2, item_view2, _ = self.generate()
        _, _, layer_user_view = self.generate()

        neg_user_view = self.users_mixup(layer_user_view, user_view2, neg_candidate_users)

        # user_scores = self.get_user_sim_score(layer_user_emb, user_view2, u_idx)
        # user_fn_mask = self.fn_detection(user_scores, self.fnk)
        # #hard negative sample mask
        # hns_scores = self.get_user_sim_score(neg_user_view, user_view2, u_idx)
        # hns_fn_mask = self.fn_detection(hns_scores,  self.fnk)
        # only user masking
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], neg_user_view[u_idx], self.temp)
        item_cl_loss = self.InfoNCE_item(item_view1[i_idx], item_view2[i_idx], self.temp)

        return user_cl_loss + item_cl_loss
    
    def InfoNCE_item(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    # FN elimination
    def InfoNCE(self, view1, view2, neg_view, temperature, b_cos=True):
        if b_cos:
            view1, view2, neg_view = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(neg_view, dim=1)
        LARGE_NUM = 1e9
        '''original cl loss'''
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        hns_score = torch.matmul(view1, neg_view.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        hns_score = torch.exp(hns_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / (hns_score + ttl_score+10e-6))

        return torch.mean(cl_loss)
    
    # FN attraction
    # def InfoNCE(self, view1, view2, mask, temperature, b_cos=True):
    #     if b_cos:
    #         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    #     LARGE_NUM = 1e9
    #     '''original cl loss'''
    #     # pos_score = (view1 * view2).sum(dim=-1)
    #     # pos_score = torch.exp(pos_score / temperature)
    #     dot_score = torch.matmul(view1, view2.transpose(0, 1))
    #     p_score = (view1 * view2).sum(dim=-1)
    #     p_score = torch.exp(p_score / temperature)
    #     pos_score = torch.exp((dot_score * mask) / temperature).mean(dim=1)
    #     pos_score = (p_score + pos_score) / 2
    #     '''FN elimination'''
    #     ttl_score = torch.exp(dot_score / temperature).sum(dim=1)
    #     # ttl_score = torch.exp((dot_score - mask*LARGE_NUM) / temperature).sum(dim=1)
    #     cl_loss = -torch.log(pos_score / ttl_score+10e-6)

    #     return torch.mean(cl_loss)
    
    
    def get_user_sim_score(self, support_user_view, user_gcn_emb, u_idx):
        '''regularization'''
        support_user_view, user_gcn_emb = F.normalize(support_user_view, dim=1), F.normalize(user_gcn_emb, dim=1)
        score = torch.matmul(support_user_view[u_idx], user_gcn_emb[u_idx].t())
        return score

    def get_item_sim_score(self, support_item_view, item_gcn_emb, i_idx):
        '''regularization'''
        support_item_view, item_gcn_emb = F.normalize(support_item_view, dim=1), F.normalize(item_gcn_emb, dim=1)
        score = torch.matmul(support_item_view[i_idx], item_gcn_emb[i_idx].t())
        return score
    
    def fn_detection(self, score, k):
        # FN이라고 판단한 node의 index를 return하는 함수
        '''top-k strategy
           정렬했을 때 상위 k개에 해당하는 index 위치에만 binary masking'''
        sorted_score, indices = torch.sort(score, 1, descending=True)
        # 자기자신 제거
        topk_mask = indices[:,1:k+1].unsqueeze(dim=-1)
        # 자기자신 포함
        # topk_mask = indices[:,:k].unsqueeze(dim=-1)
        batch_ind = torch.arange(indices.shape[0]).view([-1,1,1]).repeat([1,k,1]).cuda()
        taken_ind = torch.cat([batch_ind, topk_mask], -1).view([-1,2])
        sparse_mask = torch.sparse_coo_tensor(list(zip(*taken_ind)), torch.ones((len(taken_ind))), (score.shape[0],score.shape[1]))
        dense_mask = sparse_mask.to_dense().cuda()
        '''threshold strategy'''
        t_mask = torch.where(score>=0.9, 1, 0).cuda()
        return dense_mask * t_mask
        

