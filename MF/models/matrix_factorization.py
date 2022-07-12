from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import trange
import heapq

class MF:

    def __init__(self, sparse_matrix, hidden_dim, K, lr, beta, epochs):
        """
        Arguments
        - sparse_matrix : user-item rating matrix
        - hidden_dim : number of latent dimensions
        - K : number of recommendations
        - lr : learning rate
        - beta : regularization parameter
        - epochs : num of epochs
        """

        self.sparse_matrix = sparse_matrix.fillna(0).to_numpy()
        self.item_n, self.user_n = sparse_matrix.shape
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.beta = beta
        self.epochs = epochs

    def train(self):
        # user와 item의 latent feature 행렬 정규분포로 초기화
        # scale : 정규분포의 표준편차
        self.I = np.random.normal(scale=1./self.hidden_dim, size=(self.item_n, self.hidden_dim))
        self.U = np.random.normal(scale=1./self.hidden_dim, size=(self.user_n, self.hidden_dim))

        # bias 초기화
        self.item_bias = np.zeros(self.item_n)
        self.user_bias = np.zeros(self.user_n)
        self.total_mean = np.mean(self.sparse_matrix[np.where(self.sparse_matrix != 0)])

        # Train 데이터 생성
        idx, jdx = self.sparse_matrix.nonzero()
        train_set = list(zip(idx, jdx))

        training_log = []
        progress= trange(self.epochs, desc = 'train-rmse : nan')
        for idx in progress:
            np.random.shuffle(train_set)

            for i, u in train_set:
                # error 
                y = self.sparse_matrix[i,u]
                pred = self.predict(i, u)
                error =  y - pred
                # update bias
                self.item_bias[i] += self.lr * (error - self.beta * self.item_bias[i])
                self.user_bias[u] += self.lr * (error - self.beta * self.user_bias[u])
                # update latent factors
                # I_i = self.I[i,:]
                self.I[i, :] += self.lr * (error * self.U[u,:] - self.beta * self.I[i, :])
                self.U[u, :] += self.lr * (error * self.I[i, :] - self.beta * self.U[u, :])

            rmse = self.evaluate()
            progress.set_description('train-rmse : %0.4f' %rmse)
            progress.refresh()
            training_log.append((idx, rmse))

        self.pred_matrix = self.get_pred_matrix()

    def predict(self, i, u):
        return (
            self.total_mean
            + self.item_bias[i]
            + self.user_bias[u]
            + self.U[u,:].dot(self.I[i,:].T)
        )

    def get_pred_matrix(self):
        return (
            self.total_mean
            + self.item_bias[:,np.newaxis]
            + self.user_bias[np.newaxis:,]
            + self.I.dot(self.U.T)
        )

    def get_recommendation(self):
        pred_matrix = self.get_pred_matrix().T
        recommendations = []
        # user마다 K개 만큼의 predicted rating이 높은 아이템들의 index 구하기
        for u in range(pred_matrix.shape[0]):
            rec_idx = list(map(list(pred_matrix[u]).index, heapq.nlargest(self.K, pred_matrix[u])))
            recommendations.append(rec_idx)
        return np.array(recommendations)

    def evaluate(self):
        idx, jdx = self.sparse_matrix.nonzero()
        pred_matrix = self.get_pred_matrix()
        ys, preds = [], []
        for i, j in zip(idx, jdx):
            ys.append(self.sparse_matrix[i, j])
            preds.append(pred_matrix[i, j])

        error = mean_squared_error(ys, preds)
        return np.sqrt(error)

    def val_evaluate(self, val_set):
        pred_matrix = self.get_pred_matrix()
        ys, preds = [], []
        for i, j, rating in val_set:
            ys.append(rating)
            preds.append(pred_matrix[i, j])

        error = mean_squared_error(ys, preds)
        return np.sqrt(error)

    def test_evaluate(self, test_set):
        pred_matrix = self.get_pred_matrix()
        ys, preds = [], []
        for i, j, rating in test_set:
            ys.append(rating)
            preds.append(pred_matrix[i, j])

        error = mean_squared_error(ys, preds)
        return np.sqrt(error)

    




