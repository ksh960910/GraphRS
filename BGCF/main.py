import torch

from BGCF import BGCFLayer
from utility.load_data import Data, sampled_graph_to_matrix
from utility.parser import parse_args
from utility.batch_test import *

from time import time
import tqdm


if __name__ == '__main__':

    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id))

    t0 = time()

    data = Data(path = args.path, batch_size = args.batch_size)
    n_users, n_items = data.n_users, data.n_items
    
    print(f'#users, #items : {n_users}, {n_items}')

    # 모델 정의
    model = BGCFLayer(n_users, n_items, args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    obs_graph = Data(path = args.path, batch_size = args.batch_size)
    obs_adj_matrix, full_adj_matrix = obs_graph.get_adj_mat()
    obs_adj_matrix = torch.Tensor(obs_adj_matrix.toarray())
    full_adj_matrix = torch.Tensor(full_adj_matrix.toarray())

    print(obs_adj_matrix.shape, full_adj_matrix.shape)


    loss, mf_loss, emb_loss = 0., 0., 0.
    loss_loger, pre_loger, rec_loger, ndcg_loger = [], [], [], []

    for epoch in range(5):
        

        sampled_graph = sampled_graph_to_matrix(path = args.path, iteration = epoch, batch_size=args.batch_size)
        adj_matrix = sampled_graph.get_adj_mat().toarray()
        adj_matrix = torch.Tensor(adj_matrix)

        n_batch = n_users // args.batch_size + 1
        print(n_batch)

        for iteration in range(n_batch): #n_batch
            t1 = time()
            
            obs_users, obs_pos_items, obs_neg_items = obs_graph.sample()
            print(len(obs_users), len(obs_pos_items))

            full_users, full_pos_items, full_neg_items = obs_graph.full_sample()
            print(len(full_users), len(full_pos_items), len(full_neg_items))
            
            users, pos_items, neg_items = sampled_graph.sample(obs_users)
            # print('sampled : ', adj_matrix.shape, len(users), len(pos_items))

            '''불러오는 sampled graph matrix마다 pos, neg item set을 만들고
            bpr loss 함수에다가 각 item들에 해당하는 embedding을 입력으로 넣어줌'''

            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                        pos_items,
                                                                        neg_items,
                                                                        adj_matrix,
                                                                        obs_users,
                                                                        obs_pos_items,
                                                                        obs_neg_items,
                                                                        obs_adj_matrix,
                                                                        epoch)

            
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                            pos_i_g_embeddings,
                                                                            neg_i_g_embeddings)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            
            # 나중에 아래줄은 지울것
            print(f'iteration : {iteration+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')
        # print(f'Epoch : {epoch+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')

        if (epoch + 1) % 10 != 0:
            perf_str = f'Epoch {epoch+1}  {time() - t1 :.1f} : train loss = {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}'
            print(perf_str)
        
        if epoch%2==0:
            t2 = time()
            users_to_test = list(data.test_set.keys())
            ret = test(model, users_to_test, users, pos_items, neg_items, full_pos_items, full_neg_items, adj_matrix, full_adj_matrix, epoch)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            
        
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                        'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str) 
                