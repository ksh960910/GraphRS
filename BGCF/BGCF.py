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
        self.attention_weight = nn.Parameter(initializer(torch.empty(self.in_features, self.out_features)))

    def forward(self, adj_matrix):
            score = torch.inner(self.user_emb, self.item_emb)
            score = torch.where(adj_matrix > 0, score, 0)
            score = F.softmax(score, dim=1)
            if self.flag==True:
                # output = torch.matmul((torch.concat(score, self.item_emb)), self.attention_weight)
                output = torch.matmul(score, self.item_emb)
                output = torch.matmul(output, self.attention_weight)
                return output
            else:
                # output = torch.matmul((torch.concat(score.T, self.user_emb)), self.attention_weight)
                output = torch.matmul(score.T, self.user_emb)
                output = torch.matmul(output, self.attention_weight)
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
        self.mean_weight = nn.Parameter(initializer(torch.empty(self.in_features, self.out_features)))

    def forward(self, adj_matrix):
        self.n_users = self.user_emb.shape[0]
        self.n_items = self.item_emb.shape[0]

        neighbor_num_user, neighbor_num_item = torch.zeros(self.n_users), torch.zeros(self.n_items)
        for i in range(self.n_users):
            # neighbor_num_user[i] = sum(adj_matrix[i])
            neighbor_num_user[i] = torch.sum(adj_matrix[i])
        for j in range(self.n_items):
            # neighbor_num_item[j] = sum(adj_matrix.T[j])
            neighbor_num_item[j] = torch.sum(adj_matrix.T[j])
        # neighbor_num은 n_j에 해당하는 부분
        
        if self.flag == True:
            output = torch.matmul(adj_matrix, self.item_emb)
            for i in range(neighbor_num_user.shape[0]):
                output[i] = (1 / (neighbor_num_user[i] + 1e-8)) * output[i]
            output = torch.matmul(output, self.mean_weight)
        else:
            output = torch.matmul(adj_matrix.T, self.user_emb)
            for j in range(neighbor_num_item.shape[0]):
                output[j] = (1 / (neighbor_num_item[j] + 1e-8)) * output[j]
            output = torch.matmul(output, self.mean_weight)

        return output




class BGCFLayer(nn.Module):
    # TODO : 초기함수에 positive/negative item 리스트를 받아서 각 positive/negative item들의 embedding이 output으로 나오게끔
    def __init__(self, n_users, n_items, args):
        super(BGCFLayer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.in_features = args.embed_size      # 기존 노드의 embedding 차원수 
        self.out_features = args.layer_size    # 결과 weight의 embedding 차원수
        self.device = torch.device('cuda:' + str(args.gpu_id))


        self.node_dropout = args.node_dropout
        self.mess_dropout = eval(args.mess_dropout)[0]
        self.batch_size = args.batch_size
        self.decay = args.regs


        
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

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

        
    def create_bpr_loss(self, users, pos_items, neg_items):
        # users = torch.nan_to_num(users)
        # pos_items = torch.nan_to_num(pos_items)
        # neg_items = torch.nan_to_num(neg_items)

        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)
        
        regularizer = (torch.norm(users) ** 2
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
                obs_adj_matrix):
                
        # self.adj_matrix = self._convert_sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        self.adj_matrix = adj_matrix
        self.obs_adj_matrix = obs_adj_matrix
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

        h_tilde_2_user = self.sampled_mean_user(adj_matrix)
        h_tilde_2_item = self.sampled_mean_item(adj_matrix)
        
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
        h_tilde_obs_user = torch.tanh(self.obs_user(obs_adj_matrix))
        h_tilde_obs_item = torch.tanh(self.obs_item(obs_adj_matrix))

        
        
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

        h_tilde_user = nn.Dropout(self.mess_dropout)(h_tilde_user)
        h_tilde_pos_item = nn.Dropout(self.mess_dropout)(h_tilde_pos_item)
        h_tilde_neg_item = nn.Dropout(self.mess_dropout)(h_tilde_neg_item)
        
        h_tilde_user = F.normalize(h_tilde_user, p=2)
        h_tilde_pos_item = F.normalize(h_tilde_pos_item, p=2)
        h_tilde_neg_item = F.normalize(h_tilde_neg_item, p=2)


        return h_tilde_user, h_tilde_pos_item, h_tilde_neg_item