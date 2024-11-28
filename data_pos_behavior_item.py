import numpy as np 
import pdb 
import time
import pandas as pd
import itertools as IT
from collections import defaultdict
from itertools import combinations
from joblib import Parallel, delayed
from itertools import combinations 
import scipy
from scipy.sparse import csr_matrix 
import itertools 
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed


#和data_pos_behavior.py代码一样，只是把user_id和poi_id交互了名字

# file_name = '../data_process/core10/data_interaction_final_reid.csv'
file_name = '../data_process/core10/train.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id','photo_id','time_us','ulat','ulong','plat','plong'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')

print(data_interaction.head(5))
# 交换列名 user_id 和 poi_id
columns = data_interaction.columns.tolist()
idx1, idx2 = columns.index("user_id"), columns.index("poi_id")
columns[idx1], columns[idx2] = columns[idx2], columns[idx1]
data_interaction.columns = columns
print(data_interaction.head(5))
# pdb.set_trace()



data_interaction_u = data_interaction.drop_duplicates(subset='user_id')
u_id_max = data_interaction_u.shape[0]
data_interaction_i = data_interaction.drop_duplicates(subset='poi_id')
i_id_max = data_interaction_i.shape[0]

# 101700 81488
print(u_id_max,i_id_max)
i_id_max = 81689+1
# pdb.set_trace()

u_ilist = dict()
u_ilist_list = [set()]*u_id_max #[set() for _ in range(u_id_max)]
i_ulist = dict()
i_ulist_list = [set()]*i_id_max #[set() for _ in range(i_id_max)]

user_id_list =[]

data_interaction_u = data_interaction.groupby('user_id').agg(list).reset_index()
for index, row in data_interaction_u.iterrows():
    user_id = row['user_id']
    poi_list = row['poi_id']

    user_id_list.append(user_id)
    if user_id not in u_ilist:
        u_ilist[user_id]=set(poi_list) 
        u_ilist_list[user_id]=set(poi_list)
    else:
        print("user id double appear error",user_id)
        pdb.set_trace()


data_interaction_i = data_interaction.groupby('poi_id').agg(list).reset_index()
for index, row in data_interaction_i.iterrows():
    poi_id = row['poi_id']
    user_list = row['user_id']

    if poi_id not in i_ulist:
        i_ulist[poi_id]=set(user_list)
        i_ulist_list[poi_id]= set(user_list)
    else:
        print("poi id double appear error",poi_id)
        pdb.set_trace()

#针对任意item i和j，就计算交集，计算得到值，构建成矩阵。
alpah=0.1
# i_sim = np.zeros((len(i_ulist),len(i_ulist))) #reid 之后就可以用了。
i_sim = np.zeros((i_id_max,i_id_max))




#使用稀疏矩阵的方案。
def sets_to_sparse_matrix(sets_list):
    # 构建全集和索引
    all_elements = set().union(*sets_list)
    element_index = {element: idx for idx, element in enumerate(all_elements)}
    
    # 构造稀疏矩阵数据
    data = []
    rows = []
    cols = []
    for row_idx, s in enumerate(sets_list):
        for element in s:
            data.append(1)
            rows.append(row_idx)
            cols.append(element_index[element])
    
    # 稀疏布尔矩阵
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(sets_list)+1, len(all_elements)+1), dtype=int)
    return sparse_matrix

def intersection_lengths_sparse(sets_list):
    # 转换为稀疏矩阵
    sparse_matrix = sets_to_sparse_matrix(sets_list)
    # print('sparse_matrix is end')
    # 稀疏矩阵乘法计算交集大小
    # intersect_counts = sparse_matrix @ sparse_matrix.T
    intersect_counts = sparse_matrix.dot(sparse_matrix.T)
    # print('sparse_matrix dot is end')
    return intersect_counts.toarray()

    # pdb.set_trace()
    # # 提取上三角部分，不包括对角线元素
    # upper_triangle = np.triu_indices_from(intersect_counts.toarray(), k=1)
    # return intersect_counts[upper_triangle].toarray()

# 示例
print('set intersection set, start')
start_time = time.time()
result_iu = intersection_lengths_sparse(i_ulist_list)
elapsed_time = time.time() - start_time
print('i_ulist_list, computer the length of set and set, the time:',elapsed_time)#只要15s，最快的方法。


alpah = 0.2
# list_user_pair = list(itertools.product(range(u_id_max), range(u_id_max)))
# pos_u_v = np.zeros((len(u_ilist),len(u_ilist))) #reid 之后就可以用了。
pos_u_v = np.zeros((u_id_max+1,u_id_max+1))
result_ui = intersection_lengths_sparse(u_ilist_list)
list_user_pair = np.argwhere(result_ui>0) #6830267,2

print('list_user_pair is end,list_user_pair.shape',list_user_pair.shape)


# result_iu = 1.0/(result_iu+alpah) #太耗时，且内存支持不了，存不下。
# result_iu = result_iu+alpah
# np.reciprocal(result_iu, out=result_iu) #改了计算方式，还是太耗时，且内存支持不了，存不下。因此放弃这种方式。

#开多线程计算方式
def calculate_intersection(one_pair):
    u,v = one_pair
    same_item = u_ilist[u]&u_ilist[v] 
    same_item_list = list(same_item)
    sim_uv = np.sum(1/(result_iu[np.ix_(same_item_list, same_item_list)]+alpah)) 
    pos_u_v[v][u] = sim_uv
    pos_u_v[u][v] = sim_uv
    return sim_uv

def compute_intersections(list_user_pair, max_workers=8):
    list_sim_uv = []
    
    # 使用并行计算
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # combinations 生成唯一集合对组合，避免重复计算
        res_sim_uv = executor.map(calculate_intersection, list_user_pair)
        list_sim_uv.append(res_sim_uv)
    return list_sim_uv


start_time = time.time()
list_sim_uv = []
res_sim_uv = Parallel(n_jobs=32)(
    delayed(calculate_intersection)(pair) for pair in list_user_pair
)
list_sim_uv.extend(res_sim_uv)
elapsed_time = time.time() - start_time
print('--each pair time--',elapsed_time) #550.3678503036499, 9min

np.save('../data_process/core10/train/user_pos_pair.pkl',pos_u_v)

pdb.set_trace()



start_time = time.time()
list_sim_uv = compute_intersections(list_user_pair,64)
elapsed_time = time.time() - start_time
print('--each pair time--',elapsed_time) # 10448.213822126389 3h

np.save(pos_u_v,'../data_process/core10/train/user_pos_pair_item.pkl')

pdb.set_trace()

exit()



# ================废弃代码===========================

#常规计算方式
elapsed_time_all = 0
elapsed_time_count =0
for one_pair in list_user_pair:
    start_time = time.time()

    u,v = one_pair
    same_item = u_ilist[u]&u_ilist[v] 
    same_item_list = list(same_item)
    sim_uv = np.sum(1/(result_iu[np.ix_(same_item_list, same_item_list)]+alpah)) 
    
    pos_u_v[u][v] = sim_uv
    pos_u_v[v][u] = sim_uv

    elapsed_time = time.time() - start_time
    elapsed_time_all+=elapsed_time
    elapsed_time_count+=1
    elapsed_time_average = elapsed_time_all/elapsed_time_count
    print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average)

elapsed_time = time.time() - start_time
print('--all time--',elapsed_time,'---avg time--',elapsed_time_average) #预计22分钟。


pdb.set_trace()



exit()



# ================不要的代码===========================








#使用布尔运算的方案。
def sets_to_bool_matrix(sets_list):
    # 获取所有元素的全集
    all_elements = set().union(*sets_list)
    element_index = {element: idx for idx, element in enumerate(all_elements)}
    
    # 构造布尔矩阵，行表示集合，列表示元素
    bool_matrix = np.zeros((len(sets_list), len(all_elements)), dtype=bool)
    for row_idx, s in enumerate(sets_list):
        for element in s:
            bool_matrix[row_idx, element_index[element]] = True
            
    return bool_matrix

def intersection_lengths_matrix(sets_list):
    # 将集合列表转换为布尔矩阵
    bool_matrix = sets_to_bool_matrix(sets_list)
    
    # 使用矩阵乘法计算交集大小
    intersect_counts = bool_matrix @ bool_matrix.T
    # 取上三角部分并去除对角线元素
    upper_triangle = np.triu_indices_from(intersect_counts, k=1)
    return intersect_counts[upper_triangle]

# 示例
print('set intersection set, start')
start_time = time.time()
result = intersection_lengths_matrix(i_ulist_list)
elapsed_time = time.time() - start_time
print('--train--',elapsed_time)
pdb.set_trace()


#做并行计算的方案
def intersection_length(a, b):
    return len(a & b)

def intersection_lengths_parallel(sets_list):
    # 使用并行计算计算每对set的交集长度
    return Parallel(n_jobs=-1)(delayed(intersection_length)(a, b) for a, b in combinations(sets_list, 2))

print('set intersection set, start')
result = intersection_lengths_parallel(i_ulist_list)
print(result)  # 输出每对set的交集长度


#普通原始的方案，每个需要0.3s进行计算，一共需要23w*23w*0.3,时间不可接受。
for i in i_ulist: 
    start_time = time.time()
    print(i)
    for j in i_ulist:
       i_sim[i][j] = 1/(len(i_ulist[i]&i_ulist[j])+alpah)
       i_sim[j][i] = 1/(len(i_ulist[i]&i_ulist[j])+alpah)
    elapsed_time = time.time() - start_time
    print('--train--',elapsed_time)





