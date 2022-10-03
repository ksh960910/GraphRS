import numpy as np
from copy import deepcopy
import random
from time import time


class generate_graph(object):
    def __init__(self, path):
        self.path = path 
        train_file = path + '/train.txt'
        
        self.neighbor_dict = {}
        self.user, self.item = [], []
        
        with open(train_file, 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    self.neighbor_dict[int(l[0])] = [int(i) for i in l[1:]]
                    self.user.append(int(l[0]))
        # self.user = self.user[:128]
        
    def jaccard_index(self, u_i, u_j, neighbor_dict):
        u_i_neighbor = self.neighbor_dict[u_i]
        u_j_neighbor = self.neighbor_dict[u_j]
        
        intersection = len(list(set(u_i_neighbor).intersection(u_j_neighbor)))
        union = (len(u_i_neighbor) + len(u_j_neighbor)) - intersection
        return float(intersection) / union
        # 원래 쓴 jaccard index code -- 너무 느려서 바꿈
        # return len(list(set(u_i_neighbor) & set(u_j_neighbor))) / len(list(set(u_i_neighbor) | set(u_j_neighbor)))

    # 새로운 graph의 node j 에다가 기존 graph의 어떤 node의 neighborhood를 복사할지 zeta에 담기
    def node_copying(self, iteration):
        self.iteration = iteration

        t1 = time()
        zeta = []
        
        for u_j in self.user:
            nor = 0
            zeta_distribution = []
            for u_i in self.user:
                nor+=self.jaccard_index(u_j, u_i, self.neighbor_dict)
            for u_m in self.user:
                zeta_distribution.append(self.jaccard_index(u_j, u_m, self.neighbor_dict) / nor)
            zeta.append(random.choices(self.user, weights=zeta_distribution)[0])
        print('total node copying time cost : ', time() - t1)
        np.save(self.path + '/zeta/zeta_'+str(iteration+1)+'.npy', zeta)
        
        return zeta
    
    def generate_graph(self, zeta, epsilon, iteration):
        t2 = time()
        self.epsilon = epsilon
        self.iteration = iteration
        self.zeta = zeta
        
        generated_node = []
        
        with open(self.path+'/sampled_graph/sampled_graph_'+str(iteration+1), 'w') as f:
            for i in self.user:
                if random.uniform(0,1) < 1-self.epsilon:  # 1-epsilon의 확률로 원래 neighbor 넣기
                    generated_node.append(i)
                else:                                # epsilon의 확률로 zeta에 있는 node의 neighbor로 copy해서 넣기
                    generated_node.append(zeta[i])
                    
                # 만들어지는 새로운 graph를 txt 파일로 저장
                f.write(str(i))
                f.write(' ')
                for j in self.neighbor_dict[generated_node[i]][:-1]:
                    f.write(str(j))
                    f.write(' ')
                f.write(str(self.neighbor_dict[generated_node[i]][-1]))
                f.write('\n')
        print('#',iteration+1,' Graph sampled time cost : ', time() - t2)
