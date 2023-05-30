'''
Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
import random
import numpy as np

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
        '''XSimGCL'''
        ego_embeddings = torch.cat([user_embed, item_embed], dim=0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        all_embeddings_cl_1 = ego_embeddings
        for k in range(self.n_hops):
            ego_embeddings = torch.sparse.mm(self.interact_mat, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
            if k==self.layer_cl:
                all_embeddings_cl_1 = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.n_users, self.n_items])
        user_all_embeddings_cl_1, item_all_embeddings_cl_1 = torch.split(all_embeddings_cl_1, [self.n_users, self.n_items])

        '''perturb==False일때도 cll-1번째 layer의 embedding이 필요함'''
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl, user_all_embeddings_cl_1, item_all_embeddings_cl_1
        return user_all_embeddings, item_all_embeddings
        # if perturbed:
        #     return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        # return user_all_embeddings, item_all_embeddings


class X2SimGCL(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(X2SimGCL, self).__init__()

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
        self.beta = args_config.beta

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

    def forward(self,batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        '''perturb 되지 않은 3-hop 정보들이 aggregation된 최종 embedding'''
        rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb, cl_user_emb_add, cl_item_emb_add = self.gcn(self.user_embed,
                                                                                                        self.item_embed,
                                                                                                        perturbed = True,
                                                                                                        edge_dropout=self.edge_dropout,
                                                                                                        mess_dropout=self.mess_dropout)
        '''batch에 해당하는 user/pos/neg embedding 뽑음'''
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user], rec_item_emb[pos_item], rec_item_emb[neg_item]

        rec_loss =self.create_bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        batch_loss = rec_loss + self.l2_reg_loss(self.decay, user_emb, pos_item_emb)
        cl_loss = self.cl_rate * self.cal_cl_loss([user, pos_item], rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb, cl_user_emb_add, cl_item_emb_add)

        return batch_loss + cl_loss

    def generate(self, perturb, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                            self.item_embed,
                                            perturbed=perturb,
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
    
    # '''negative sampling의 output으로는 batch가 적용된 embedding말고
    #    전체 [29858,64], [40981,64] 사이즈의 embedding이 나와야함'''
    # # batch 적용 버전
    # def users_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_users, u_idx):
    #     '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
    #        neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
    #     pos_gcn_emb = pos_gcn_emb[u_idx]
    #     # print('pos gcn emb :', pos_gcn_emb.shape)
    #     neg_candidate_users = np.array(neg_candidate_users)
    #     u_idx = u_idx.tolist()
    #     batch_neg_candidate_users = neg_candidate_users[u_idx]
    #     # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
    #     neg_emb = neg_gcn_emb[batch_neg_candidate_users] # [n_users, n_negs, emb_size]
    #     # print('neg emb :', neg_emb.shape)

    #     '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
    #     neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
    #     scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]
    #     # neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],:,:]

    #     return neg_emb

    # def items_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_items, i_idx):
    #     '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
    #        neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
    #     pos_gcn_emb = pos_gcn_emb[i_idx]
    #     neg_candidate_items = np.array(neg_candidate_items)
    #     i_idx = i_idx.tolist()
    #     batch_neg_candidate_items = neg_candidate_items[i_idx]
    #     # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
    #     neg_emb = neg_gcn_emb[batch_neg_candidate_items] # [n_users, n_negs, emb_size]
        
    #     '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
    #     neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
    #     scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]
    #     # neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],:,:]

    #     return neg_emb

    # '''perturb된 l* layer와 perturb된 final layer의 mixing'''
    # def users_mixup(self, user_view, neg_candidate_users):
    #     '''처음으로 l*에서의 embedding들이 쭈욱 들어옴
    #        들어온 애들 가지고 i와 다른 애들을 서로 mixing'''
    #     neg_emb = user_view[neg_candidate_users] # [n_users, n_negs, emb_size] 각 유저마다 n_negs개만큼 뽑힌 user의 embedding
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     neg_emb = alpha*user_view.unsqueeze(dim=1) + (1-alpha)*neg_emb
    #     scores = (user_view.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]
    #     return neg_emb

    # def items_mixup(self, item_view, neg_candidate_items):
    #     '''처음으로 l*에서의 embedding들이 쭈욱 들어옴
    #        들어온 애들 가지고 i와 다른 애들을 서로 mixing'''
    #     neg_emb = item_view[neg_candidate_items] # [n_users, n_negs, emb_size] 각 유저마다 n_negs개만큼 뽑힌 user의 embedding
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     neg_emb = alpha*item_view.unsqueeze(dim=1) + (1-alpha)*neg_emb
    #     scores = (item_view.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]
    #     return neg_emb


    # '''negative sampling의 output으로는 batch가 적용된 embedding말고
    #    전체 [29858,64], [40981,64] 사이즈의 embedding이 나와야함'''
    # def users_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_users):
    #     '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
    #        neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
        
    #     # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
    #     neg_emb = neg_gcn_emb[neg_candidate_users] # [n_users, n_negs, emb_size]

    #     '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
    #     neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
    #     scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]

    #     return neg_emb

    # def items_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_items):
    #     '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
    #        neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
        
    #     # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
    #     neg_emb = neg_gcn_emb[neg_candidate_items] # [n_users, n_negs, emb_size]
        
    #     '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
    #     alpha = torch.rand_like(neg_emb).cuda()
    #     # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
    #     neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
    #     scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]

    #     return neg_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2, user_view3, item_view3):
        '''view1은 perturb된 최종 embedding, view2는 perturb된 l*-layer embedding'''
        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
        u_idx = torch.unique(torch.LongTensor(idx[0].cpu()).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.LongTensor(idx[1].cpu()).type(torch.long)).to(self.device)
        
        '''perturb 되지 않은 embedding
           cl_view는 perturb되지 않은 l*-layer에서의 embedding'''
        # user_view0, item_view0, cl_user_view0, cl_item_view0 = self.generate(perturb=False)

        '''perturb 되지 않은 positive embedding과 perturb된 negative embedding mixup하기
           cl_user_view0는 perturb되지 않은 l*-layer의 embedding
           user_view2는 perturb된 l*-layer의 embedding'''
        # neg_user_view = self.users_mixup(user_view2, neg_candidate_users)
        # neg_item_view = self.items_mixup(item_view2, neg_candidate_items)
        # user_view2, item_view2 = self.mixup(user_view0, user_view1), self.mixup(item_view0, item_view1)

        # neg_user_view, neg_item_view = neg_user_view.mean(dim=1), neg_item_view.mean(dim=1)
        # neg_user_view, neg_item_view = neg_user_view.squeeze(dim=1), neg_item_view.squeeze(dim=1)


        '''perturb 된 positive embedding과 perturb된 negative embedding mixup하기
           view1은 perturbed 3-hop aggregation된 애
           view2는 perturbed l*번째 layer까지 aggregation된 애'''
        # user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], neg_user_view, self.temp)
        # item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], neg_item_view, self.temp)
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], user_view3[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], item_view3[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def InfoNCE(self, view1, view2, view3, temperature, b_cos=True):
        if b_cos:
            view1, view2, view3 = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(view3, dim=1)
        '''서로 다른 view의 자기자신 노드끼리 내적값을 구할 수 있도록 element-wise multiplication'''
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        '''view1.unsqueeze(dim=2) -> [batch_size, emb_size, 1]
           neg_view               -> [batch_size, emb_size, n_negs]'''
        t_score = torch.matmul(view1, view2.transpose(0,1))
        t_score = torch.exp(t_score / temperature).sum(dim=1)
        ttl_score = torch.matmul(view1, view3.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))
        return torch.mean(cl_loss)
        
        # ttl_score = (view1.unsqueeze(dim=1)*neg_view).sum(dim=1)
        # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        # # print('ttl :', ttl_score)
        # t_score = torch.matmul(view1, view2.transpose(0,1))
        # t_score = torch.exp(t_score / temperature).sum(dim=1)
        # # print('t : ', t_score)
        # cl_loss = -torch.log(pos_score / (t_score+ttl_score + 10e-6))
        # # cl_loss = -torch.log(pos_score / t_score + 10e-6)
        # return torch.mean(cl_loss)


        '''0.8 elementwise + 0.2 matmul'''
        # t_score = torch.matmul(view1, view2.transpose(0,1))
        # t_score = torch.exp(t_score / temperature).sum(dim=1)
        # ttl_score = torch.mul(view1, neg_view)
        # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        # cl_loss = -torch.log(pos_score / (self.beta*ttl_score + (1-self.beta)*t_score+10e-6))

        '''서로 다른 view의 자신과 다른 노드끼리 내적값을 구할 수 있도록 matmul'''
        '''cl loss 분모에 mixed 된거로 matmul한 걸 더하는 방법 (batch개만큼 더하는 꼴)'''
        # t_score = torch.matmul(view1, view2.transpose(0,1))
        # t_score = torch.exp(t_score / temperature).sum(dim=1)
        # ttl_score = torch.matmul(view1, neg_view.transpose(0, 1))
        # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        # cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))

        '''cl loss 분모에 mixed 된거로 element wise한 걸 더하는 방법 (1개만 더하는 꼴)'''
        # ttl_score = torch.matmul(view1, view2.transpose(0,1))
        # t_score = (view1 * neg_view)
        # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        # t_score = torch.exp(t_score / temperature).sum(dim=1)
        # cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))
        

    # def InfoNCE(self, view1, view2, temperature, b_cos=True):
    #     if b_cos:
    #         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    #     '''서로 다른 view의 자기자신 노드끼리 내적값을 구할 수 있도록 element-wise multiplication'''
    #     pos_score = (view1 * view2).sum(dim=-1)
    #     pos_score = torch.exp(pos_score / temperature)
    #     '''서로 다른 view의 자신과 다른 노드끼리 내적값을 구할 수 있도록 matmul'''
    #     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    #     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    #     return torch.mean(cl_loss)


