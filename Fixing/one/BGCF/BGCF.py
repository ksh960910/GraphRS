import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class AttenConv(nn.Module):
    def __init__(self, user_emb, item_emb, args, flag=True):
        super(AttenConv, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.in_features = args.embed_size
        self.out_features = args.layer_size
        self.flag = flag

        initializer = nn.init.xavier_uniform_
        self.attention_weight = nn.Parameter(initializer(torch.empty(self.in_features , self.out_features)))
        # self.attention_weight = nn.Parameter(initializer(torch.empty(self.in_features * 2 , self.out_features)))

    def forward(self, adj_matrix):
            # 가장 최근, 0.1~ 0.4에서 계속 왔다갔다
            # if self.flag == True:
            #     e_j = torch.matmul(self.user_emb, self.attention_weight) # 6040 64
            #     e_k = torch.sparse.mm(adj_matrix.t(), self.user_emb) # 3090 64
            #     e_k = torch.matmul(e_k, self.attention_weight) # 3090 64
            #     score = torch.matmul(e_j, e_k.t()) # 6040 3090
            #     score = F.softmax(score, dim=1)

            #     output = torch.matmul(score, e_k) # 6040 64
            #     output = torch.matmul(output, self.attention_weight)

            #     return output

            # else:
            #     e_j = torch.matmul(self.item_emb, self.attention_weight) # 3090 64
            #     e_k = torch.sparse.mm(adj_matrix, self.item_emb) # 6040 64
            #     e_k = torch.matmul(e_k, self.attention_weight) # 6040 64 
            #     score = torch.matmul(e_j, e_k.t()) # 3090 6040
            #     score = F.softmax(score, dim=1)

            #     output = torch.matmul(score, e_k)
            #     output = torch.matmul(output, self.attention_weight)
                
            #     return output

            # if self.flag==True:
            #     score = torch.sparse.mm(adj_matrix.t(), self.user_emb) # 3090 64
            #     score = torch.matmul(score, self.attention_weight)
            #     atten = torch.inner(self.user_emb, score) # 6040 3090
            #     atten = F.softmax(atten, dim=1)

            #     output = torch.matmul(atten, score) # 6040 64
            #     output = torch.matmul(output, self.attention_weight)
            #     return output

            # else:
            #     score = torch.sparse.mm(adj_matrix, self.item_emb) # 6040 64
            #     score = torch.matmul(score, self.attention_weight)
            #     atten = torch.inner(self.item_emb, score) # 3090 6040
            #     atten = F.softmax(atten, dim=1)

            #     output = torch.matmul(atten, score) # 3090 64
            #     output = torch.matmul(output, self.attention_weight)
            #     return output

        e_j = torch.matmul(self.user_emb, self.attention_weight)  # 6040 64
        e_k = torch.matmul(self.item_emb, self.attention_weight)  # 3900 64
        u_neigh = torch.sparse.mm(adj_matrix, e_k)  # 6040 64
        i_neigh = torch.sparse.mm(adj_matrix.t(), e_j)  # 3900 64
        if self.flag == True:
            dot = torch.inner(u_neigh, i_neigh)
            atten = F.softmax(dot, dim=1)

            output = torch.matmul(atten, e_k)  # 6040 64
            output = torch.matmul(output, self.attention_weight)  # 6040 64

        else:
            dot = torch.inner(i_neigh, u_neigh)
            atten = F.softmax(dot, dim=1)  # 3900 6040

            output = torch.matmul(atten, e_j)  # 3900 64
            output = torch.matmul(output, self.attention_weight)  # 3900 64

        return output

class MeanConv(nn.Module):
    def __init__(self, user_emb, item_emb, args, flag=True):
        super(MeanConv, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.in_features = args.embed_size
        self.out_features = args.layer_size
        self.flag = flag

        initializer = nn.init.xavier_uniform_
        self.mean_weight = nn.Parameter(initializer(torch.empty(self.in_features, self.out_features)), requires_grad=True)
        # self.mean_weight = nn.Parameter(initializer(torch.empty(self.in_features * 2 , self.out_features)), requires_grad=True)

    def forward(self, adj_matrix, user_n_j, item_n_j):
        # if self.flag==False:
        #     e_k = torch.sparse.mm(adj_matrix.t(), self.user_emb)
        #     e_k = torch.matmul(e_k, self.mean_weight) # 3090 64
        #     e_k = torch.mul(e_k, item_n_j)
        #     output = torch.matmul(e_k, self.mean_weight)
        #     return output
        # else:
        #     e_k = torch.sparse.mm(adj_matrix, self.item_emb)
        #     e_k = torch.matmul(e_k, self.mean_weight)
        #     e_k = torch.mul(e_k, user_n_j)
        #     output = torch.matmul(e_k, self.mean_weight)
        #     return output

        # 지금까지 쓰던거
        # if self.flag==True:
        #     e_k = torch.sparse.mm(adj_matrix, self.item_emb) #6040 64
        #     e_k = torch.matmul(e_k, self.mean_weight)
        #     e_k = torch.mul(e_k, user_n_j)
        #     # e_k = e_k / e_k.shape[0]
        #     # e_k = torch.mean(e_k, dim=1)
        #     output = torch.matmul(e_k, self.mean_weight)
        #     # output = torch.matmul(torch.concat((self.user_emb, e_k), dim=1), self.mean_weight)
        #     return output
        # else:
        #     e_k = torch.sparse.mm(adj_matrix.t(), self.user_emb)
        #     e_k = torch.matmul(e_k, self.mean_weight)
        #     e_k = torch.mul(e_k, item_n_j)
        #     # e_k = e_k / e_k.shape[0]
        #     # e_k = torch.mean(e_k, dim=1)
        #     output = torch.matmul(e_k, self.mean_weight)
        #     # output = torch.matmul(torch.concat((self.item_emb, e_k), dim=1), self.mean_weight)
        #     return output

        if self.flag==True:
            e_k = torch.matmul(self.item_emb, self.mean_weight)
            u_neigh = torch.sparse.mm(adj_matrix, e_k)
            norm = torch.mul(u_neigh, user_n_j)
            output = torch.matmul(norm, self.mean_weight)

        else:
            e_k = torch.matmul(self.user_emb, self.mean_weight)
            i_neigh = torch.sparse.mm(adj_matrix.t(), e_k)
            norm = torch.mul(i_neigh, item_n_j)
            output = torch.matmul(norm, self.mean_weight)

        return output



class BGCFLayer(nn.Module):
    # TODO : 초기함수에 positive/negative item 리스트를 받아서 각 positive/negative item들의 embedding이 output으로 나오게끔
    def __init__(self, n_users, n_items, args):
        super(BGCFLayer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.in_features = args.embed_size      # 기존 노드의 embedding 차원수 
        self.out_features = args.layer_size    # 결과 weight의 embedding 차원수
        self.device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
        self.path = args.path
        self.dataset = args.dataset


        self.node_dropout = eval(args.node_dropout)[0]
        self.mess_dropout = eval(args.mess_dropout)[0]
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.l2 = args.l2


        
        initializer = nn.init.xavier_uniform_
        
        # self.W = nn.Parameter(initializer(torch.empty(in_features, out_features)))
        # W_1은 a*e aggregate 부분, W_2는 e aggregate 부분
        # self.weight_dict = nn.ParameterDict({
        #     'W_1_user' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features))),
        #     'W_1_item' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features))),
        #     'W_2_user' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features))),
        #     'W_2_item' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features))),
        #     'W_obs_user' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features))),
        #     'W_obs_item' : nn.Parameter(initializer(torch.empty(self.in_features, self.out_features)))
        # })

        initializer = nn.init.xavier_uniform_
        self.user_emb = initializer(torch.empty(self.n_users, self.out_features)).to(self.device)
        self.item_emb = initializer(torch.empty(self.n_items, self.out_features)).to(self.device)

        self.sampled_attention_user = AttenConv(self.user_emb, self.item_emb, args, flag=True)
        self.sampled_attention_item = AttenConv(self.user_emb, self.item_emb, args, flag=False)
        self.sampled_mean_user = MeanConv(self.user_emb, self.item_emb, args, flag=True)
        self.sampled_mean_item = MeanConv(self.user_emb, self.item_emb, args, flag=False)
        self.obs_user = MeanConv(self.user_emb, self.item_emb, args, flag=True)
        self.obs_item = MeanConv(self.user_emb, self.item_emb, args, flag=False)
        
        # self.embedding_dict = nn.ParameterDict({
        #     'user_emb' : nn.Parameter(initializer(torch.empty(self.n_users, self.out_features))),
        #     'item_emb' : nn.Parameter(initializer(torch.empty(self.n_items, self.out_features)))
        # })

    # def count_neighbor(self, adj_matrix):
    #     adj_matrix = torch.from_numpy(adj_matrix.toarray())
    #     neighbor_num_user, neighbor_num_item = torch.zeros(adj_matrix.shape[0]), torch.zeros(adj_matrix.shape[1])
    #     user_n_j = torch.ones(adj_matrix.shape[0], self.in_features)
    #     item_n_j = torch.ones(adj_matrix.shape[1], self.in_features)
    #     for i in range(adj_matrix.shape[0]):
    #         # neighbor_num_user[i] = sum(adj_matrix[i])
    #         neighbor_num_user[i] = torch.sum(adj_matrix[i])
    #         if neighbor_num_user[i] != 0:
    #             user_n_j[i] = (user_n_j[i] / neighbor_num_user[i])
    #         else:
    #             user_n_j[i] = 1e-8
            
    #     for j in range(adj_matrix.shape[1]):
    #         # neighbor_num_item[j] = sum(adj_matrix.T[j])
    #         neighbor_num_item[j] = torch.sum(adj_matrix.T[j])
    #         if neighbor_num_item[j] != 0:
    #             item_n_j[j] = (item_n_j[j] / neighbor_num_item[j])
    #         else:
    #             item_n_j[j] = 1e-8

    #     return user_n_j, item_n_j

    def get_count_neighbor(self, X, iteration):
        try:
            final_user_n_j = torch.from_numpy(np.load(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_user_neighbor.npy'))
            final_item_n_j = torch.from_numpy(np.load(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_item_neighbor.npy'))

        except:
            final_user_n_j, final_item_n_j = self.count_neighbor(X)
            final_user_n_j = final_user_n_j.numpy()
            final_item_n_j = final_item_n_j.numpy()
            np.save(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_user_neighbor', final_user_n_j)
            np.save(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_item_neighbor', final_item_n_j)
            final_user_n_j = torch.from_numpy(np.load(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_user_neighbor.npy'))
            final_item_n_j = torch.from_numpy(np.load(self.path + self.dataset + '/sample_graph_'+str(iteration)+'_item_neighbor.npy'))

        return final_user_n_j, final_item_n_j



    def count_neighbor(self, X):
        n_users, n_items = X.shape[0], X.shape[1]
        coo = X.tocoo()
        user_n_j, item_n_j = [], []
        final_user_n_j = torch.ones(n_users, self.in_features)
        final_item_n_j = torch.ones(n_items, self.in_features)

        r = coo.row
        c = coo.col

        for i in range(n_users):
            user_n_j.append(1 / len(r[r==i]))
            # final_user_n_j[i] *= user_n_j[i]
        for j in range(n_items):
            if len(c[c==j])!=0:
                item_n_j.append(1/ len(c[c==j]))
            else:
                item_n_j.append(1/ 1e+8)
            # final_item_n_j[j] *= item_n_j[j]
        final_user_n_j = torch.Tensor(user_n_j).repeat(self.in_features,1).T
        final_item_n_j = torch.Tensor(item_n_j).repeat(self.in_features,1).T

        return final_user_n_j, final_item_n_j

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

        
    def create_bpr_loss(self, users, pos_items, neg_items):
        # users = torch.nan_to_num(users)
        # pos_items = torch.nan_to_num(pos_items)
        # neg_items = torch.nan_to_num(neg_items)

        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        # mf_loss = -1 * torch.mean(maxi)
        mf_loss = -1 * torch.sum(maxi)

        # regularizer = (torch.norm(users) ** 2
        #                        + torch.norm(pos_items) ** 2
        #                        + torch.norm(neg_items) ** 2) / 2
        
        regularizer = self.l2 * (torch.norm(users) ** 2
                               + torch.norm(pos_items) ** 2
                               + torch.norm(neg_items) ** 2) / 2
        
        emb_loss = self.decay * regularizer / self.batch_size
        
        return mf_loss+emb_loss, mf_loss, emb_loss
            
    
    def forward(self,  
                users,
                pos_items, 
                neg_items,
                adj_matrix,
                obs_users,
                obs_pos_items, 
                obs_neg_items, 
                obs_adj_matrix,
                iteration
                ):
        sample_user_n_j, sample_item_n_j = self.get_count_neighbor(adj_matrix, iteration)
        obs_user_n_j, obs_item_n_j = self.get_count_neighbor(obs_adj_matrix, 0)
                
        adj_matrix = self._convert_sp_mat_to_sp_tensor(adj_matrix)
        obs_adj_matrix = self._convert_sp_mat_to_sp_tensor(obs_adj_matrix)

        adj_matrix = self.sparse_dropout(adj_matrix, self.node_dropout, adj_matrix._nnz()).to(self.device)
        obs_adj_matrix = self.sparse_dropout(obs_adj_matrix, self.node_dropout, obs_adj_matrix._nnz()).to(self.device)


        sample_user_n_j = sample_user_n_j.to(self.device)
        sample_item_n_j = sample_item_n_j.to(self.device)
        obs_user_n_j = obs_user_n_j.to(self.device)
        obs_item_n_j = obs_item_n_j.to(self.device)

        h_tilde_1_user = self.sampled_attention_user(adj_matrix)
        h_tilde_1_item = self.sampled_attention_item(adj_matrix)

        h_tilde_2_user = self.sampled_mean_user(adj_matrix, sample_user_n_j, sample_item_n_j)
        h_tilde_2_item = self.sampled_mean_item(adj_matrix, sample_user_n_j, sample_item_n_j)
        
        
        h_tilde_sampled_user = torch.cat((h_tilde_1_user, h_tilde_2_user), dim=1)
        h_tilde_sampled_item = torch.cat((h_tilde_1_item, h_tilde_2_item), dim=1)

        # h_tilde_sampled_user = F.normalize(h_tilde_sampled_user, p=4)
        # h_tilde_sampled_item = F.normalize(h_tilde_sampled_item, p=4)

        
        h_tilde_sampled_user = h_tilde_sampled_user[users,:]
        h_tilde_sampled_pos_item = h_tilde_sampled_item[pos_items,:]
        h_tilde_sampled_neg_item = h_tilde_sampled_item[neg_items,:]
        
        h_tilde_obs_user = torch.tanh(self.obs_user(obs_adj_matrix, obs_user_n_j, obs_item_n_j))
        h_tilde_obs_item = torch.tanh(self.obs_item(obs_adj_matrix, obs_user_n_j, obs_item_n_j))

        # h_tilde_obs_user = F.normalize(h_tilde_obs_user, p=4)
        # h_tilde_obs_item = F.normalize(h_tilde_obs_item, p=4)
        
        h_tilde_obs_user = h_tilde_obs_user[obs_users,:]
        h_tilde_obs_pos_item = h_tilde_obs_item[obs_pos_items,:]
        h_tilde_obs_neg_item = h_tilde_obs_item[obs_neg_items,:]

        # Final embedding
        h_tilde_user = torch.tanh(torch.cat((h_tilde_sampled_user, h_tilde_obs_user), dim=1))
        h_tilde_pos_item = torch.tanh(torch.cat((h_tilde_sampled_pos_item, h_tilde_obs_pos_item), dim=1))
        h_tilde_neg_item = torch.tanh(torch.cat((h_tilde_sampled_neg_item, h_tilde_obs_neg_item), dim=1))
        
        h_tilde_user = F.normalize(h_tilde_user, p=5)
        h_tilde_pos_item = F.normalize(h_tilde_pos_item, p=5)
        h_tilde_neg_item = F.normalize(h_tilde_neg_item, p=5)


        return h_tilde_user, h_tilde_pos_item, h_tilde_neg_item