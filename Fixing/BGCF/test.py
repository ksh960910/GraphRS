import torch
import torch.nn as nn
import sys
from main import *

if __name__ == '__main__':
    # a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    # b = torch.Tensor([[2,3,4],[5,6,7],[8,2,5]])

    # print(torch.inner(a,b))
    # print(torch.mm(a,b))
    # print(torch.sum(a * b, dim=-1))
    n_users, n_items = 6040, 3953

    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    obs_graph = Data(path = args.path+args.dataset, batch_size = args.batch_size)
    obs_adj_matrix, test_adj_mat = obs_graph.get_adj_mat()
    print(type(obs_adj_matrix))

    for i in range(5):
        
    # for i in range(2):
    #     if i==0:
    #         obs_adj_matrix, test_adj_mat = obs_graph.get_adj_mat()
    #         x = obs_adj_matrix
    #     else:
    #         obs_adj_matrix, test_adj_mat = obs_graph.get_adj_mat()
    #         x = np.concatenate([x, obs_adj_matrix])
    # print(len(x))
    

    # n = []
    # m = []
    # def count_n(X):
    #     coo = X.tocoo()
    #     r = coo.row
    #     c = coo.col

    #     for i in range(n_users):
    #         n.append(len(r[r==i]))
    #     for j in range(n_items):
    #         if len(c[c==j])!=0:
    #             m.append(len(c[c==j]))
    #         else:
    #             m.append(1e+8)
        
    #     return m
    # # count_n(obs_adj_matrix)
    # n = count_n(obs_adj_matrix)
    # x = torch.Tensor(n).repeat(64,1).T
    # print(x)
    # print(x.shape)
    # obs_adj_matrix = torch.from_numpy(obs_adj_matrix.toarray())
    # obs_adj_matrix = _convert_sp_mat_to_sp_tensor(obs_adj_matrix).cuda()
    # obs_adj_matrix = torch.from_numpy(obs_adj_matrix.toarray()).cuda()
    # print(sys.getsizeof(obs_adj_matrix.storage()))

    initializer = nn.init.xavier_uniform_
    user_emb = initializer(torch.empty(6040, 64))
    # print('user : ', torch.cuda.memory_reserved())
    item_emb = initializer(torch.empty(3953, 64))
    # print('item : ', torch.cuda.memory_reserved())

    # score = torch.exp(torch.inner(user_emb, item_emb))
    # print(torch.cuda.memory_reserved())

    # # score = torch.exp(torch.inner(user_emb, item_emb))
    # print('score : ', torch.cuda.memory_reserved())
    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())
    # coef = torch.zeros(n_users, n_items)


    # def get_attention_coefficient(user_emb, item_emb, adj_matrix):
    #     score = torch.exp(torch.inner(user_emb, item_emb))
    #     score = torch.multiply(score, adj_matrix)
    #     # score = score.mul(self.adj_matrix)
        
    #     for i in range(score.shape[0]):
    #         norm = torch.sum(score[i])
    #         if norm==0:
    #             print(norm)
    #         # norm += 1e-6
    #         normalized_score = score[i] / norm
    #         coef[i] = normalized_score

    #     return coef

    # coef = get_attention_coefficient(user_emb, item_emb, obs_adj_matrix)
    # print(coef[0])

    # # def get_attention_fix(source_emb, target_emb, adj_matrix):

    # #     e = torch.matmul(source_emb, )


    # sco = torch.ones(n_users, n_items, requires_grad=True)
    # score = torch.inner(user_emb, item_emb)
    # print(score)
    # # score = torch.where(obs_adj_matrix > 0, score, obs_adj_matrix)

    # # score = torch.multiply(score, obs_adj_matrix)
    # score = nn.functional.softmax(score, dim=-1)
    # print(score[0])


    # print(user_emb)
    # user_emb[0][:3]*=100
    # print(user_emb)
    # print(nn.functional.softmax(user_emb, dim=1))

    # for i in range(2):
    #     a = 13
    #     for j in range(5):
    #         b = 5
    #         print(j)
    #     print(a*j)

        
    # print(torch.matmul(coef, item_emb).shape)
    # print(torch.mul(coef, item_emb))

    # a = torch.Tensor([[1,0,0],[0,1,1]])
    # print(a)
    # b = torch.Tensor([[1,2,1],[2,1,2],[1,1,2]])
    # print(torch.matmul(a,b))
    # print(torch.mul(a,b))
    
    # tr = sp.load_npz('Data/ml-1m/obs_adj_mat.npz')
    # print(tr[0])
    # print('====================================================')
    # t = sp.load_npz('Data/ml-1m/test_adj_mat.npz')
    # print(t[0])



    