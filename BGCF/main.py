import torch

from BGCF import BGCFLayer
from utility.load_data import Data, sampled_graph_to_matrix
from utility.parser import parse_args
from utility.batch_test import *
from utility.helper import early_stopping

from time import time
import tqdm
import sys


if __name__ == '__main__':

    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id))

    t0 = time()

    data = Data(path = args.path+args.dataset, batch_size = args.batch_size)
    n_train, n_users, n_items = data.n_train, data.n_users, data.n_items
    
    print(f'#users, #items, #train items : {n_users}, {n_items}, {n_train}')

    # 모델 정의
    model = BGCFLayer(n_users, n_items, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    obs_graph = Data(path = args.path+args.dataset, batch_size = args.batch_size)
    obs_adj_matrix, test_adj_mat = obs_graph.get_adj_mat()
    obs_adj_matrix = torch.from_numpy(obs_adj_matrix.toarray())
    test_adj_mat = torch.from_numpy(test_adj_mat.toarray())

    # print(obs_adj_matrix.shape, test_adj_mat.shape)

    loss, mf_loss, emb_loss = 0., 0., 0.
    loss_loger, pre_loger, rec_loger, ndcg_loger = [], [], [], []
    cur_best_pre_0, stopping_step = 0, 0


    for epoch in range(args.epoch): # args.epoch
        t1 = time()

        sampled_graph = sampled_graph_to_matrix(path = args.path+args.dataset, iteration = epoch, batch_size=args.batch_size)
        adj_matrix = sampled_graph.get_adj_mat()
        adj_matrix = torch.from_numpy(adj_matrix.toarray())
        # adj_matrix = torch.Tensor(adj_matrix).cuda()

        n_batch = n_train // args.batch_size + 1
        print('Number of batch : ', n_batch)

        # for name, param in model.named_parameters():
        #     print(name, param)

        for iteration in range(n_batch): #n_batch
            with torch.autograd.set_detect_anomaly(True):
                obs_users, obs_pos_items, obs_neg_items = obs_graph.sample()
                
                # test할 때 필요한 전체 
                # full_users, full_pos_items, full_neg_items = obs_graph.full_sample()
                # print(len(full_users), len(full_pos_items), len(full_neg_items))
                
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
                                                                            obs_adj_matrix)

                # u_g_embeddings /= args.epoch
                # pos_i_g_embeddings /= args.epoch
                # neg_i_g_embeddings /= args.epoch

                
                batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                pos_i_g_embeddings,
                                                                                neg_i_g_embeddings)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss+=batch_loss
                mf_loss+=batch_mf_loss
                emb_loss+=batch_emb_loss
            
            # 나중에 아래줄은 지울것
            # print(f'iteration : {iteration+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')
        # print(f'Epoch : {epoch+1} Train time : {time() - t1:.2f} train loss : {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}')

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and (epoch+1) % args.verbose == 0:
                perf_str = f'Epoch {epoch+1}  Train time {(time() - t1)//60 :.0f}min {(time() - t1)%60 :.0f}sec | train loss = {loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f}'
                print('Train: ', perf_str)
            continue
        
        t2 = time()
        users_to_test = list(data.test_set.keys())
        s_users_to_test = list(sampled_graph.train_items.keys())
        # ret = test(model, users_to_test, s_users_to_test, neg_items, adj_matrix, test_adj_mat, epoch)
        ret = test(model, users_to_test, s_users_to_test, neg_items, obs_neg_items, adj_matrix, test_adj_mat)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        
    
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                    'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch+1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1], ret['ndcg'][0], ret['ndcg'][-1])
        print('Test : ', perf_str) 

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@ %dmin %.0fsec \trecall=[%s], precision=[%s], ndcg=[%s]" % \
                (idx, (time() - t0)//60, (time() - t0) % 60, '\t'.join(['%.5f' % r for r in recs[idx]]),
                '\t'.join(['%.5f' % r for r in pres[idx]]),
                '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
            