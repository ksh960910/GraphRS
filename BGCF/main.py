import torch

from BGCF import BGCFLayer
from utility.load_data import Data, sampled_graph_to_matrix
from utility.parser import parse_args

from time import time
import tqdm


if __name__ == '__main__':

    args = parse_args()
    # 완성형은 GATLayer(6040, 3953, 64,64, pos_items, neg_items)
    model = BGCFLayer(1000, 3953, args)

    t0 = time()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    path = 'ml-1m'
    obs_graph = Data(path = path, batch_size = args.batch_size)
    obs_adj_matrix = obs_graph.get_adj_mat().toarray()
    obs_adj_matrix = torch.Tensor(obs_adj_matrix)
    for epoch in range(1):
        loss, mf_loss, emb_loss = 0., 0., 0.

        obs_users, obs_pos_items, obs_neg_items = obs_graph.sample()
        
        sampled_graph = sampled_graph_to_matrix(path = path, iteration = epoch, batch_size=1000)
        adj_matrix = sampled_graph.get_adj_mat().toarray()
        adj_matrix = torch.Tensor(adj_matrix)
        users, pos_items, neg_items = sampled_graph.sample()
        
        for iteration in range(1):
            t1 = time()

            '''불러오는 sampled graph matrix마다 pos, neg item set을 만들고
            bpr loss 함수에다가 각 item들에 해당하는 embedding을 입력으로 넣어줌'''

            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(pos_items,
                                                                        neg_items,
                                                                        adj_matrix,
                                                                        obs_pos_items,
                                                                        obs_neg_items,
                                                                        obs_adj_matrix,
                                                                        iteration)
            
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                            pos_i_g_embeddings,
                                                                            neg_i_g_embeddings)
            
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            
            print(f'train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')

            # if (epoch + 1) % 10 != 0:
            #     perf_str = f'Epoch {epoch}  {time() - t1 :.1f} : train loss == {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}'
            #     print(perf_str)
                
            t2 = time()
            