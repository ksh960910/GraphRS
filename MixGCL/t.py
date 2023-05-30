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

tensor = torch.ones([10,32,64])
tensor = torch.rand_like(tensor)
print(tensor)
print(tensor.shape)
t = time()
print(tensor[[i for i in range(tensor.shape[0])],:,:])
print('time : ', time() - t)
print(tensor.shape)
    # '''negative sampling의 output으로는 batch가 적용된 embedding말고
    #    전체 [29858,64], [40981,64] 사이즈의 embedding이 나와야함'''
    # def users_mixup(self, pos_gcn_emb, neg_gcn_emb, neg_candidate_users):
    #     '''pos_gcn_emb는 perturbed되지 않은 user / item 3-hop aggregation embedding
    #        neg_gcn_emb는 self.generate()에서 나온 user / item view1'''
        
    #     # neg_candidate = self.get_neg_candidate(neg_gcn_emb) # [n_users, n_negs, emb_size]
    #     neg_emb = neg_gcn_emb[neg_candidate_users] # [n_users, n_negs, emb_size]

    #     '''neg_emb size와 같은 0~1까지의 random number가 담긴 tensor 생성'''
    #     alpha = (torch.rand_like(neg_emb)/2).cuda()
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
    #     alpha = (torch.rand_like(neg_emb)/2).cuda()
    #     # print('pos unsqu : ', pos_gcn_emb.unsqueeze(dim=1).shape)
    #     neg_emb = alpha*pos_gcn_emb.unsqueeze(dim=1) + (1-alpha)*neg_emb  # [n_users, n_negs, emb_size]
        
    #     scores = (pos_gcn_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_emb = neg_emb[[i for i in range(neg_emb.shape[0])],indices,:]

    #     return neg_emb

    # def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2, neg_candidate_users, neg_candidate_items):
    #     '''view1은 perturb된 최종 embedding, view2는 perturb된 l*-layer embedding'''
    #     # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
    #     # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
    #     u_idx = torch.unique(torch.LongTensor(idx[0].cpu()).type(torch.long)).to(self.device)
    #     i_idx = torch.unique(torch.LongTensor(idx[1].cpu()).type(torch.long)).to(self.device)
        
    #     '''perturb 되지 않은 embedding
    #        cl_view는 perturb되지 않은 l*-layer에서의 embedding'''
    #     user_view0, item_view0, cl_user_view0, cl_item_view0 = self.generate(perturb=False)

    #     '''perturb 되지 않은 positive embedding과 perturb된 negative embedding mixup하기
    #        cl_user_view0는 perturb되지 않은 l*-layer의 embedding
    #        user_view2는 perturb된 l*-layer의 embedding'''
    #     # neg_user_view = self.users_mixup(user_view0, user_view1, neg_candidate_users)
    #     # neg_item_view = self.items_mixup(item_view0, item_view1, neg_candidate_items)
    #     neg_user_view = self.users_mixup(cl_user_view0, user_view2, neg_candidate_users)
    #     neg_item_view = self.items_mixup(cl_item_view0, item_view2, neg_candidate_items)
    #     # neg_user_view = self.users_mixup(user_view2, neg_candidate_users)
    #     # neg_item_view = self.items_mixup(item_view2, neg_candidate_items)
    #     # user_view2, item_view2 = self.mixup(user_view0, user_view1), self.mixup(item_view0, item_view1)

    #     # neg_user_view, neg_item_view = neg_user_view.mean(dim=1), neg_item_view.mean(dim=1)
    #     # neg_user_view, neg_item_view = neg_user_view.squeeze(dim=1), neg_item_view.squeeze(dim=1)


    #     '''perturb 된 positive embedding과 perturb된 negative embedding mixup하기
    #        view1은 perturbed 3-hop aggregation된 애
    #        view2는 perturbed l*번째 layer까지 aggregation된 애'''
    #     # user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], neg_user_view, self.temp)
    #     # item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], neg_item_view, self.temp)
    #     user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], neg_user_view[u_idx], self.temp)
    #     item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], neg_item_view[i_idx], self.temp)
    #     return user_cl_loss + item_cl_loss

    # def InfoNCE(self, view1, view2, neg_view, temperature, b_cos=True):
    #     if b_cos:
    #         view1, view2, neg_view = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(neg_view, dim=1)
    #     '''서로 다른 view의 자기자신 노드끼리 내적값을 구할 수 있도록 element-wise multiplication'''
    #     pos_score = (view1 * view2).sum(dim=-1)
    #     pos_score = torch.exp(pos_score / temperature)

    #     '''view1.unsqueeze(dim=2) -> [batch_size, emb_size, 1]
    #        neg_view               -> [batch_size, emb_size, n_negs]'''
    #     t_score = torch.matmul(view1, view2.transpose(0,1))
    #     t_score = torch.exp(t_score / temperature).sum(dim=1)
    #     ttl_score = torch.matmul(view1, neg_view.transpose(0, 1))
    #     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))
    #     return torch.mean(cl_loss)
        
    #     # ttl_score = (view1.unsqueeze(dim=1)*neg_view).sum(dim=1)
    #     # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     # # print('ttl :', ttl_score)
    #     # t_score = torch.matmul(view1, view2.transpose(0,1))
    #     # t_score = torch.exp(t_score / temperature).sum(dim=1)
    #     # # print('t : ', t_score)
    #     # cl_loss = -torch.log(pos_score / (t_score+ttl_score + 10e-6))
    #     # # cl_loss = -torch.log(pos_score / t_score + 10e-6)
    #     # return torch.mean(cl_loss)


    #     '''0.8 elementwise + 0.2 matmul'''
    #     # t_score = torch.matmul(view1, view2.transpose(0,1))
    #     # t_score = torch.exp(t_score / temperature).sum(dim=1)
    #     # ttl_score = torch.mul(view1, neg_view)
    #     # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     # cl_loss = -torch.log(pos_score / (self.beta*ttl_score + (1-self.beta)*t_score+10e-6))

    #     '''서로 다른 view의 자신과 다른 노드끼리 내적값을 구할 수 있도록 matmul'''
    #     '''cl loss 분모에 mixed 된거로 matmul한 걸 더하는 방법 (batch개만큼 더하는 꼴)'''
    #     # t_score = torch.matmul(view1, view2.transpose(0,1))
    #     # t_score = torch.exp(t_score / temperature).sum(dim=1)
    #     # ttl_score = torch.matmul(view1, neg_view.transpose(0, 1))
    #     # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     # cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))

    #     '''cl loss 분모에 mixed 된거로 element wise한 걸 더하는 방법 (1개만 더하는 꼴)'''
    #     # ttl_score = torch.matmul(view1, view2.transpose(0,1))
    #     # t_score = (view1 * neg_view)
    #     # ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    #     # t_score = torch.exp(t_score / temperature).sum(dim=1)
    #     # cl_loss = -torch.log(pos_score / (t_score+ttl_score+10e-6))
        

    # # def InfoNCE(self, view1, view2, temperature, b_cos=True):
    # #     if b_cos:
    # #         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    # #     '''서로 다른 view의 자기자신 노드끼리 내적값을 구할 수 있도록 element-wise multiplication'''
    # #     pos_score = (view1 * view2).sum(dim=-1)
    # #     pos_score = torch.exp(pos_score / temperature)
    # #     '''서로 다른 view의 자신과 다른 노드끼리 내적값을 구할 수 있도록 matmul'''
    # #     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    # #     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    # #     cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    # #     return torch.mean(cl_loss)


