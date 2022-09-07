import numpy as np
import random
import scipy.sparse as sp
from time import time
from utility.node_copying import generate_graph

# observed data로 sparse matrix 및 adjacency matrix 만들기
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        
        train_file = path + '/train.txt'
        # path : ml-1m
        
        self.n_users, self.n_items = 0, 0
        self.exist_users = []
        with open(train_file, 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))

        self.n_users+=1
        self.n_items+=1
        print(self.n_users, self.n_items)

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items = {}

        with open(train_file, 'r') as f:
            for l in f.readlines():
                if len(l)==0:
                    break
                l = l.strip('\n').split(' ')
                uid, items = int(l[0]), [int(i) for i in l[1:]]

                for i in items:
                    self.R[uid, i] = 1

                self.train_items[uid] = items
                
    def get_adj_mat(self):
        try:
            obs_adj_mat = sp.load_npz(self.path + '/obs_adj_mat.npz')
            
        except Exception:
            obs_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/obs_adj_mat.npz', obs_adj_mat)
            
        return obs_adj_mat
    
    
    def create_adj_mat(self):
        obs_adj_mat = self.R.todok()[:1000,:]
        print('already create observed graph adjacency matrix', obs_adj_mat.shape)
        return obs_adj_mat.tocsr()
    
    # bgcf는 G_obs로부터 만들어진 sampled graphs에 대해 x hat들의 integral을 구함
    def sample(self):
        # positive / negative items 나누기
        if self.batch_size <= self.n_users:
            obs_users = random.sample(self.exist_users, self.batch_size)
        else:
            obs_users = [random.choice(self.exist_users) for _ in range(self.batch_size)]
            
        def sample_pos_items_for_u(u, num):
            # u유저의 neighbor중 num개 만큼 positive item sampling
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # u유저의 neighbor가 아닌 item 중 num개 만큼 sampling
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

    #         def sample_neg_items_for_u_from_pools(u, num):
    #             neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
    #             return random.sample(neg_items, num)     

        obs_pos_items, obs_neg_items = [], []
        for u in obs_users:
            obs_pos_items += sample_pos_items_for_u(u,1)
            obs_neg_items += sample_neg_items_for_u(u,1)

        return obs_users, obs_pos_items, obs_neg_items


# node copying으로 만들어진 graph들에 대해서 adj matrix 만들고 npz 저장하는 과정 
class sampled_graph_to_matrix(object):
    def __init__(self, path, iteration, batch_size):
        self.path =path
        self.iteration = iteration
        self.batch_size = batch_size

        try:
            sampled_graph = self.path + '/sampled_graph/sampled_graph_' + str(iteration+1)

        except Exception:
            try:
                zeta = np.load(self.path + '/zeta/zeta_' + str(iteration+1) + '.npy')
                sampled_graph = generate_graph.generate_graph(zeta=zeta, epsilon=0.01, iteration=iteration)
            
            except Exception:
                zeta = generate_graph.node_copying(iteration = iteration)
                sampled_graph = generate_graph.generate_graph(zeta=zeta, epsilon=0.01, iteration=iteration)

        
        self.n_users, self.n_items = 0, 0
        self.exist_users = []
        self.neg_pools = {}
        
        with open(sampled_graph, 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))

        self.n_users+=1
        self.n_items+=1
        print(self.n_users, self.n_items)

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items = {}

        with open(sampled_graph, 'r') as f:
            for l in f.readlines():
                if len(l)==0:
                    break
                l = l.strip('\n').split(' ') 
                uid, items = int(l[0]), [int(i) for i in l[1:]]

                for i in items:
                    self.R[uid, i] = 1

                self.train_items[uid] = items
                
    def get_adj_mat(self):
        try:
            adj_mat = sp.load_npz(self.path + '/s_adj_mat_' + str(self.iteration+1) + '.npz')
            
        except Exception:
            adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat_' + str(self.iteration+1) + '.npz', adj_mat)
            
        return adj_mat
    
     
    def create_adj_mat(self):
        adj_mat = self.R.todok()
        print('already create adjacency matrix', adj_mat.shape)
        return adj_mat.tocsr()
    
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [random.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)
    
    # bgcf는 G_obs로부터 만들어진 sampled graphs에 대해 x hat들의 integral을 구함
    def sample(self):
        # positive / negative items 나누기
        if self.batch_size <= self.n_users:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]
            
        def sample_pos_items_for_u(u, num):
            # u유저의 neighbor중 num개 만큼 positive item sampling
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]
                
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        
        def sample_neg_items_for_u(u, num):
            # u유저의 neighbor가 아닌 item 중 num개 만큼 sampling
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items 

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u,1)
            neg_items += sample_neg_items_for_u(u,1)
            
        return users, pos_items, neg_items
