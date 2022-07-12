import numpy as np
from process_data import movielens_to_matrix
import pandas as pd

def make_binary(x):
    if x>=3.5:
        x = 1
    else:
        x = 0
    return x

# precision, recall은 binary output에만 사용 --> rating을 relevant/not relevant로 이진화 시켜줌
def precision_at_k(recommendations, ground_truth):
    # 추천된 item 중 relevant한 아이템의 개수 / 추천한 아이템의 개수
    # ground truth = ratings
    K = recommendations.shape[1]
    relevant_truth = ground_truth.copy()
    # relevant_truth['relevance'] = ground_truth['rating'].apply(make_binary)
    # relevant_truth = relevant_truth.groupby('user_id').apply(lambda x : pd.Series(x['relevance'].values, index=x['movie_id'])).unstack().fillna(0).to_numpy()
    relevant_truth = relevant_truth.unstack().apply(make_binary).unstack().to_numpy()
    n_users, n_items = relevant_truth.shape
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = relevant_truth[user_idx, item_idx].reshape((n_users, K))
    relevance = np.asfarray(relevance.sum(axis=1) / K)
    precision_k = relevance.sum() / n_users
    return precision_k

def recall_at_k(recommendations, ground_truth): 
    # 추천된 item 중 relevant한 아이템의 개수 / 전체 relevant한 아이템의 개수
    # ground_truth = test_sparse_matrix
    relevant_truth = ground_truth.copy()
    relevant_truth = relevant_truth.unstack().apply(make_binary).unstack().to_numpy()
    relevance_num = relevant_truth.sum(axis=1)
    # outlier_idx = relevant한 item이 0개인 user의 인덱스 (0으로 나눌 수 없기 때문)
    outlier_idx = np.where(relevance_num==0)
    # outlier를 지운 matrix와 recommendation
    relevant_truth = np.delete(relevant_truth, outlier_idx, axis=0)
    recommendations = np.delete(recommendations, outlier_idx, axis=0)
    K = recommendations.shape[1]
    n_users, n_items = relevant_truth.shape
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance_num = relevant_truth.sum(axis=1)
    relevance = np.array(relevant_truth[user_idx, item_idx].reshape((n_users, K)))
    relevance = np.asfarray(relevance.sum(axis=1) / relevance_num)
    recall_k = relevance.sum() / n_users
    return recall_k
