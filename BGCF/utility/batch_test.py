import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import heapq

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path = args.path, batch_size = args.batch_size)
USER_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def test(model, users_to_test, obs_adj_matrix, adj_matrix, drop_flag=False, batch_test_flag=False):
    result = {'result': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)), 'auc' : 0. }

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start:end]

        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                # 수정필요
                # u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(user_batch,
                #                                                                pos_items,
                #                                                                [],
                #                                                                adj_matrix,
                #                                                                user_batch,
                #                                                                item_batch,
                #                                                                [],
                #                                                                obs_adj_matrix,
                #                                                                iteration)
