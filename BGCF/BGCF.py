import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

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
        
            # if self.flag==True:
            #     score = torch.sparse.mm(adj_matrix, self.item_emb)
            #     score = torch.inner(score, self.item_emb)
            #     # score = torch.where(adj_matrix > 0, score, 0)
            #     score = F.softmax(score, dim=1)
                
            #     coef = torch.matmul(score, self.item_emb)
            #     # output = torch.matmul(coef, self.attention_weight)
            #     output = torch.matmul(torch.concat((coef, self.user_emb), dim=1), self.attention_weight)

            #     return output
                
            # else:
            #     score = torch.sparse.mm(adj_matrix.t(), self.user_emb)
            #     score = torch.inner(score, self.user_emb)
            #     # score = torch.sparse.mm(adj_matrix, score)
            #     # score - torch.where(adj_matrix.t() > 0, score, 0)
            #     score = F.softmax(score, dim=1)

            #     coef = torch.matmul(score, self.user_emb)
            #     # output = torch.matmul(coef, self.attention_weight)
            #     output = torch.matmul(torch.concat((coef, self.item_emb), dim=1), self.attention_weight)

            #     return output
            
            if self.flag==True:
                e_k = torch.sparse.mm(adj_matrix.t(), self.user_emb)
                e_k = torch.matmul(e_k, self.attention_weight)
                e_j = torch.matmul(self.user_emb, self.attention_weight)
                score = torch.inner(e_j, e_k)
                score = F.softmax(score, dim=1)
                output = torch.matmul(score, self.item_emb)
                output = torch.matmul(output, self.attention_weight)
                return output
            else:
                e_k = torch.sparse.mm(adj_matrix, self.item_emb)
                e_k = torch.matmul(e_k, self.attention_weight)
                e_j = torch.matmul(self.item_emb, self.attention_weight)
                score = torch.inner(e_j, e_k)
                score = F.softmax(score, dim=1)
                output = torch.matmul(score, self.user_emb)
                output = torch.matmul(output, self.attention_weight)
                return output

            # if self.flag==True:
            #     score = torch.sparse.mm(adj_matrix.t(), self.user_emb)
            #     score = torch.inner(self.user_emb, score)
            #     score = F.softmax(score, dim=1)
            #     output = torch.matmul(score, self.item_emb)
            #     output = torch.matmul(output, self.attention_weight)
            #     return output
            # else:
            #     score = torch.sparse.mm(adj_matrix, self.item_emb)
            #     score = torch.inner(score, self.item_emb)
            #     score = F.softmax(score, dim=1)
            #     output = torch.matmul(score.T, self.user_emb)
            #     output = torch.matmul(output, self.attention_weight)
            #     return output

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
        if self.flag==True:
            e_k = torch.sparse.mm(adj_matrix, self.item_emb) 
            e_k = torch.mul(e_k, user_n_j)
            # e_k = e_k / e_k.shape[0]
            # e_k = torch.mean(e_k, dim=1)
            output = torch.matmul(e_k, self.mean_weight)
            # output = torch.matmul(torch.concat((self.user_emb, e_k), dim=1), self.mean_weight)
            return output
        else:
            e_k = torch.sparse.mm(adj_matrix.t(), self.user_emb)
            e_k = torch.mul(e_k, item_n_j)
            # e_k = e_k / e_k.shape[0]
            # e_k = torch.mean(e_k, dim=1)
            output = torch.matmul(e_k, self.mean_weight)
            # output = torch.matmul(torch.concat((self.item_emb, e_k), dim=1), self.mean_weight)
            return output


        # self.n_users = self.user_emb.shape[0]
        # self.n_items = self.item_emb.shape[0]

        # neighbor_num_user, neighbor_num_item = torch.zeros(self.n_users), torch.zeros(self.n_items)
        # for i in range(self.n_users):
        #     # neighbor_num_user[i] = sum(adj_matrix[i])
        #     neighbor_num_user[i] = torch.sum(adj_matrix[i])
        # for j in range(self.n_items):
        #     # neighbor_num_item[j] = sum(adj_matrix.T[j])
        #     neighbor_num_item[j] = torch.sum(adj_matrix.T[j])
        # # neighbor_num은 n_j에 해당하는 부분
        
        # if self.flag == True:
        #     output = torch.sparse.mm(adj_matrix, self.item_emb)
        #     output = torch.mul(output, user_n_j)
        #     output = torch.matmul(output, self.mean_weight)
        #     # output = torch.matmul(torch.concat((output, self.user_emb), dim=1), self.mean_weight)
        # else:
        #     output = torch.sparse.mm(adj_matrix.t(), self.user_emb)
        #     output = torch.mul(output, item_n_j)
        #     output = torch.matmul(output, self.mean_weight)
        #     # output = torch.matmul(torch.concat((output, self.item_emb), dim=1), self.mean_weight)

        # return output




class BGCFLayer(nn.Module):
    # TODO : 초기함수에 positive/negative item 리스트를 받아서 각 positive/negative item들의 embedding이 output으로 나오게끔
    def __init__(self, n_users, n_items, args):
        super(BGCFLayer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.in_features = args.embed_size      # 기존 노드의 embedding 차원수 
        self.out_features = args.layer_size    # 결과 weight의 embedding 차원수
        self.device = torch.device('cuda:' + str(args.gpu_id))


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
        self.user_emb = initializer(torch.empty(self.n_users, self.out_features))
        self.item_emb = initializer(torch.empty(self.n_items, self.out_features))

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
        i = torch.LongTensor([coo.row, coo.col])
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
        
        regularizer = self.l2 * (torch.norm(users) ** 2
                               + torch.norm(pos_items) ** 2
                               + torch.norm(neg_items) ** 2) / 2
        
        emb_loss = self.decay * regularizer / self.batch_size
        
        return mf_loss+emb_loss, mf_loss, emb_loss
            
    
    def forward(self,  
                users,
                pos_items, 
                neg_items,
                sample_user_n_j,
                sample_item_n_j, 
                adj_matrix,
                obs_users,
                obs_pos_items, 
                obs_neg_items, 
                obs_user_n_j,
                obs_item_n_j,
                obs_adj_matrix):
                
        # sample_neighbor_num_user, sample_neighbor_num_item = self.count_neighbor(adj_matrix)
        # obs_neighbor_num_user, obs_neighbor_num_item = self.count_neighbor(obs_adj_matrix)
        # 좀전에 고친부분
        # sample_user_n_j, sample_item_n_j = self.count_neighbor(adj_matrix)
        # obs_user_n_j, obs_item_n_j = self.count_neighbor(obs_adj_matrix)
        adj_matrix = self._convert_sp_mat_to_sp_tensor(adj_matrix)
        obs_adj_matrix = self._convert_sp_mat_to_sp_tensor(obs_adj_matrix)

        adj_matrix = self.sparse_dropout(adj_matrix, self.node_dropout, adj_matrix._nnz())
        obs_adj_matrix = self.sparse_dropout(obs_adj_matrix, self.node_dropout, obs_adj_matrix._nnz())

        # self.adj_matrix = self._convert_sp_mat_to_sp_tensor(adj_matrix)
        # torch.cuda.empty_cache()
        # self.obs_adj_matrix = self._convert_sp_mat_to_sp_tensor(obs_adj_matrix)
        # self.iteration = iteration
        # coef = torch.zeros(self.n_users, self.n_items)
        
        # # score는 각각 user-item 간의 attention score
        # score = torch.exp(torch.inner(self.user_emb, self.item_emb))
        # score = torch.multiply(score, self.adj_matrix)
        # # score = score.mul(self.adj_matrix)
        
        # for i in range(score.shape[0]):
        #     norm = torch.sum(score[i])
        #     # norm += 1e-6
        #     normalized_score = score[i] / norm
        #     coef[i] = normalized_score

        # coef에다가 normalized된 최종적인 attention score 저장
        # w_1_user = self.weight_dict['W_1_user']
        # w_1_item = self.weight_dict['W_1_item']
        # w_2_user = self.weight_dict['W_2_user']
        # w_2_item = self.weight_dict['W_2_item']

        # coef = coef.cuda()
        # # h_tilde_1은 w_1과 곱해지는 부분의 embedding (e_k)를 user와 item으로 쪼개서 계산후 concat
        # h_tilde_1_user = torch.matmul(coef, self.item_emb)
        # h_tilde_1_user = torch.matmul(h_tilde_1_user, w_1_user)
        # h_tilde_1_item = torch.matmul(coef.T, self.user_emb)
        # h_tilde_1_item = torch.matmul(h_tilde_1_item, w_1_item)

        h_tilde_1_user = self.sampled_attention_user(adj_matrix)
        h_tilde_1_item = self.sampled_attention_item(adj_matrix)

        h_tilde_2_user = self.sampled_mean_user(adj_matrix, sample_user_n_j, sample_item_n_j)
        h_tilde_2_item = self.sampled_mean_item(adj_matrix, sample_user_n_j, sample_item_n_j)
        
        # neighbor_num_user, neighbor_num_item = torch.zeros(self.n_users), torch.zeros(self.n_items)
        # for i in range(self.n_users):
        #     # neighbor_num_user[i] = sum(adj_matrix[i])
        #     neighbor_num_user[i] = torch.sum(self.adj_matrix[i])
        # for j in range(self.n_items):
        #     # neighbor_num_item[j] = sum(adj_matrix.T[j])
        #     neighbor_num_item[j] = torch.sum(self.adj_matrix.T[j])
        # # neighbor_num은 n_j에 해당하는 부분
        
        # # h_tilde_2는 w_2와 곱해지는 부분
        # h_tilde_2_user = torch.matmul(self.adj_matrix, self.item_emb)
        # # h_tilde_2_user = torch.matmul(neighbor_num_user, h_tilde_2_user)
        # # (1 / 각 user의 neighbor의 수) 만큼 item embedding에 곱해주는 과정
        # for i in range(neighbor_num_user.shape[0]):
        #     h_tilde_2_user[i] = (1 / (neighbor_num_user[i] + 1e-6)) * h_tilde_2_user[i]
        # h_tilde_2_user = torch.matmul(h_tilde_2_user, w_2_user)
        # h_tilde_2_item = torch.matmul(self.adj_matrix.T, self.user_emb)
        # # h_tilde_2_item = torch.matmul(neighbor_num_item, h_tilde_2_item)
        # # 각 item의 neighbor의 수 만큼 user embedding에 곱해주는 과정
        # for j in range(neighbor_num_item.shape[0]):
        #     h_tilde_2_item[j] = (1 / (neighbor_num_item[j] + 1e-6)) * h_tilde_2_item[j]
        # h_tilde_2_item = torch.matmul(h_tilde_2_item, w_2_item)
        # # h_tilde_2 = torch.cat((h_tilde_2_user, h_tilde_2_item), dim=0)
        
        h_tilde_sampled_user = torch.cat((h_tilde_1_user, h_tilde_2_user), dim=1)
        h_tilde_sampled_item = torch.cat((h_tilde_1_item, h_tilde_2_item), dim=1)

        
        h_tilde_sampled_user = h_tilde_sampled_user[users,:]
        h_tilde_sampled_pos_item = h_tilde_sampled_item[pos_items,:]
        h_tilde_sampled_neg_item = h_tilde_sampled_item[neg_items,:]
        

        ### h_tilde_observed 
        # w_obs_user = self.weight_dict['W_obs_user']
        # w_obs_item = self.weight_dict['W_obs_item']
        h_tilde_obs_user = torch.tanh(self.obs_user(obs_adj_matrix, obs_user_n_j, obs_item_n_j))
        h_tilde_obs_item = torch.tanh(self.obs_item(obs_adj_matrix, obs_user_n_j, obs_item_n_j))

        
        
        # # observed graph의 neighbor 정보 
        # obs_neighbor_num_user, obs_neighbor_num_item = torch.zeros(self.n_users), torch.zeros(self.n_items)
        # # 각 원소 = 각 node의 neighbor수
        # for i in range(self.n_users):
        #     # obs_neighbor_num_user[i] = sum(obs_adj_matrix[i])
        #     obs_neighbor_num_user[i] = torch.sum(self.obs_adj_matrix[i])
        # for j in range(self.n_items):
        #     # obs_neighbor_num_item[j] = sum(obs_adj_matrix.T[j])
        #     obs_neighbor_num_item[j] = torch.sum(self.obs_adj_matrix.T[j])
        # # neighbor_num은 n_j에 해당하는 부분
        
        # # h_tilde_obs_user
        # h_tilde_obs_user = torch.matmul(self.obs_adj_matrix, self.item_emb)
        # for i in range(obs_neighbor_num_user.shape[0]):
        #     h_tilde_obs_user[i] = (1 / (obs_neighbor_num_user[i] + 1e-6)) * h_tilde_obs_user[i]
        # h_tilde_obs_user = torch.sigmoid(torch.matmul(h_tilde_obs_user, w_obs_user))
        
        # # h_tilde_obs_item
        # h_tilde_obs_item = torch.matmul(self.obs_adj_matrix.T, self.user_emb)
        # for j in range(obs_neighbor_num_item.shape[0]):
        #     h_tilde_obs_item[j] = (1 / (obs_neighbor_num_item[j] + 1e-6)) * h_tilde_obs_item[j]
        # h_tilde_obs_item = torch.sigmoid(torch.matmul(h_tilde_obs_item, w_obs_item))
        
        h_tilde_obs_user = h_tilde_obs_user[obs_users,:]
        h_tilde_obs_pos_item = h_tilde_obs_item[obs_pos_items,:]
        h_tilde_obs_neg_item = h_tilde_obs_item[obs_neg_items,:]

        # # Final embedding
        # h_tilde_user = torch.cat((h_tilde_sampled_user, h_tilde_obs_user), dim=1)
        # h_tilde_pos_item = torch.cat((h_tilde_sampled_pos_item, h_tilde_obs_pos_item), dim=1)
        # if h_tilde_obs_neg_item.shape[0] !=0:
        #     h_tilde_neg_item = torch.cat((h_tilde_sampled_neg_item, h_tilde_obs_neg_item), dim=1)
        # else:
        #     h_tilde_neg_item = h_tilde_sampled_neg_item

        # h_tilde_user = nn.Tanh()(h_tilde_user)
        # h_tilde_pos_item = nn.Tanh()(h_tilde_pos_item)
        # h_tilde_neg_item = nn.Tanh()(h_tilde_neg_item)
        
        # Final embedding
        h_tilde_user = torch.tanh(torch.cat((h_tilde_sampled_user, h_tilde_obs_user), dim=1))
        h_tilde_pos_item = torch.tanh(torch.cat((h_tilde_sampled_pos_item, h_tilde_obs_pos_item), dim=1))
        h_tilde_neg_item = torch.tanh(torch.cat((h_tilde_sampled_neg_item, h_tilde_obs_neg_item), dim=1))
        # if h_tilde_obs_neg_item.shape[0] !=0:
        #     h_tilde_neg_item = torch.tanh(torch.cat((h_tilde_sampled_neg_item, h_tilde_obs_neg_item), dim=1))
        # else:
        #     h_tilde_neg_item = torch.tanh(h_tilde_sampled_neg_item)

        # h_tilde_user = nn.Dropout(self.mess_dropout)(h_tilde_user)
        # h_tilde_pos_item = nn.Dropout(self.mess_dropout)(h_tilde_pos_item)
        # h_tilde_neg_item = nn.Dropout(self.mess_dropout)(h_tilde_neg_item)
        
        h_tilde_user = F.normalize(h_tilde_user, p=2)
        h_tilde_pos_item = F.normalize(h_tilde_pos_item, p=2)
        h_tilde_neg_item = F.normalize(h_tilde_neg_item, p=2)


        return h_tilde_user, h_tilde_pos_item, h_tilde_neg_item