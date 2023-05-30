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

user = torch.Tensor([[1,2,3], [10,20,30], [3,7,12]])
user1 = torch.Tensor([[10,100,1000],[1,1,1], [3,3,7]])
print(user)
# item = torch.Tensor([[2,3,4],[3,4,5]])
item = torch.Tensor([[2,3,4],[3,4,5],[4,5,6]])
print(item)
# tensor = torch.rand_like(tensor)
# tt = torch.rand_like(tt)
x = torch.matmul(user, item.t())
y = torch.matmul(user1, item.t())
# x = x.reshape(-1,1)
z = torch.stack((x, y))
print(z)
k, _ = z.max(dim=0)
print(k)
t_mask = k.ge(250)
sort_k, indices = torch.sort(k, 1, descending=True)
print(sort_k, indices)
topk_mask = indices[:,:2]
print(topk_mask)
j = torch.ones((3,3))
topk_mask = topk_mask.unsqueeze(dim=-1)
batch_ind = torch.arange(3).view([-1,1,1]).repeat([1,2,1])
taken_ind = torch.cat([batch_ind, topk_mask], -1).view([-1,2])
print(taken_ind)
print('list zip : ', list(zip(taken_ind)))
print(' * : ', list(zip(*taken_ind)))
sparse_ind = torch.sparse_coo_tensor(list(zip(*taken_ind)), torch.ones((len(taken_ind))), (3,3))
dense_ind = sparse_ind.to_dense()
print(dense_ind)
final_mask = dense_ind * t_mask
print(final_mask)
a = (final_mask==1).nonzero()
print(a)
print(a.sum(dim=1))


# print(batch_ind)


# topk_inds_expanded = tf.expand_dims(topk_inds, -1)
# batch_ind = tf.tile(
#     tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, topk, 1])
# taken_ind = tf.reshape(
#     tf.concat([batch_ind, topk_inds_expanded], -1), [-1, 2])
# taken_ind = tf.cast(taken_ind, tf.int64)
# sparse_mask = tf.sparse.SparseTensor(
#     taken_ind, tf.ones([tf.shape(taken_ind)[0]]),
#     [batch_size, tf.shape(splitted_list_large_concat)[0]])
# dense_mask_topk = tf.sparse.to_dense(sparse_mask, validate_indices=False)

# l, indices = torch.sort(k, 1, descending=True)
# a, idx = l[:,:2], indices[:,:2]
# print(idx)
# print(x)
# b = x[:,idx][:,0,:]
# print(b)
# print(l)
# m = torch.where(a>=135, True, False)
# final = m * idx
# print(final)
# for i in range(a.shape[0]):
#     print(a[i,final[i][i]])





# python main.py --dataset gowalla --gnn m1 --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 3 --context_hops 3 --pool mean --ns mixgcf --K 1


# def get_user_sim_score(self, support_user_view, user_gcn_emb, idx):
#         u_idx = torch.unique(torch.LongTensor(idx.cpu()).type(torch.long)).to(self.device)
#         # support view : [n_user, emb_size]
#         # item_gcn_emb : [n_item, emb_size]
#         score = torch.matmul(support_user_view[u_idx], user_gcn_emb[u_idx].t())
#         score, indices = torch.sort(score, 1, descending=True) # Exclude dot product score of itself
#         return score[:,1].reshape(-1,1), indices[:,1].reshape(-1,1)

# def get_item_sim_score(self, support_item_view, item_gcn_emb, idx):
#     i_idx = torch.unique(torch.LongTensor(idx.cpu()).type(torch.long)).to(self.device)
#     score = torch.matmul(support_item_view[i_idx], item_gcn_emb[i_idx].t())
#     score, indices = torch.sort(score, 1, descending=True)
#     return score[:,1].reshape(-1,1), indices[:,1].reshape(-1,1)


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


