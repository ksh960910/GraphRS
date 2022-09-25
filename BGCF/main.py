import torch

from BGCF import BGCFLayer
from utility.load_data import Data, sampled_graph_to_matrix
from utility.parser import parse_args

from time import time
import tqdm


if __name__ == '__main__':

    args = parse_args()

    t0 = time()

    data = Data(path = args.path, batch_size = args.batch_size)
    n_users, n_items = data.n_users, data.n_items
    users_to_test = list(data.test_set.keys())
    print(f'#users, #items : {n_users}, {n_items}')

    # 모델 정의
    model = BGCFLayer(n_users, n_items, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    obs_graph = Data(path = args.path, batch_size = args.batch_size)
    obs_adj_matrix = obs_graph.get_adj_mat().toarray()
    obs_adj_matrix = torch.Tensor(obs_adj_matrix)
    print(obs_adj_matrix.shape)

    for epoch in range(1):
        loss, mf_loss, emb_loss = 0., 0., 0.

        sampled_graph = sampled_graph_to_matrix(path = args.path, iteration = epoch, batch_size=args.batch_size)
        adj_matrix = sampled_graph.get_adj_mat().toarray()
        adj_matrix = torch.Tensor(adj_matrix)

        n_batch = n_users // args.batch_size + 1
        print(n_batch)

        for iteration in range(1): #n_batch
            t1 = time()
            
            # sample하는 부분이 따로따로 말고 같이 이뤄져야함 ---- 수정필요
            obs_users, obs_pos_items, obs_neg_items = obs_graph.sample()
            print(len(obs_users), len(obs_pos_items))
            
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
                                                                        iteration)

            
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                            pos_i_g_embeddings,
                                                                            neg_i_g_embeddings)
            
            print(batch_loss, batch_mf_loss, batch_emb_loss)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            
            # 나중에 아래줄은 지울것
            print(f'iteration : {iteration+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')
        print(f'Epoch : {epoch+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')

        # if (epoch + 1) % 10 != 0:
        #     perf_str = f'Epoch {epoch}  {time() - t1 :.1f} : train loss = {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}'
        #     print(perf_str)
            
        # t2 = time()
        # ret = 
            