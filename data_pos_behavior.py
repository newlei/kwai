import numpy as np 
import pdb 
import time
import pandas as pd
import itertools as IT
from collections import defaultdict

file_name = '../data_process/core10/data_interaction_final.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')

u_ilist = dict()
i_ulist = dict()
i_ulist_list = []
user_id_list =[]
u_id_max = 0
i_id_max = 0

u_id_current=0
i_id_current=0

data_interaction_u = data_interaction.groupby('user_id').agg(list).reset_index()
for index, row in data_interaction_u.iterrows():
    user_id = row['user_id']
    poi_list = row['poi_id']
    #reid 
    user_id = u_id_current
    u_id_current+=1

    user_id_list.append(user_id)
    if u_id_max<user_id:
        u_id_max = user_id
    if user_id not in u_ilist:
        u_ilist[user_id]=set(poi_list)
        i_ulist_list.append(set(poi_list))
    else:
        print("user id double appear error",user_id)
        pdb.set_trace()


data_interaction_i = data_interaction.groupby('poi_id').agg(list).reset_index()
for index, row in data_interaction_i.iterrows():
    poi_id = row['poi_id']
    user_list = row['user_id']
    #reid 
    poi_id = i_id_current
    i_id_current+=1

    if i_id_max<poi_id:
        i_id_max = poi_id
    if poi_id not in i_ulist:
        i_ulist[poi_id]=set(user_list)
    else:
        print("poi id double appear error",poi_id)
        pdb.set_trace()

#针对任意item i和j，就计算交集，计算得到值，构建成矩阵。
alpah=0.1
# i_sim = np.zeros((len(i_ulist),len(i_ulist))) #reid 之后就可以用了。
i_sim = np.zeros((i_id_max+1,i_id_max+1))


# pairs = IT.combinations(i_ulist, 2)
# nt = lambda a, b: i_ulist[a].intersection(i_ulist[b])
# res = dict([ (t, nt(*t)) for t in pairs ])


# 构建反向索引
element_to_sets = defaultdict(list)
for i, s in enumerate(i_ulist):
    for elem in s:
        element_to_sets[elem].append(i)

for target_set in i_ulist: 
    # 仅检查 target_set 中的元素对应的集合
    start_time = time.time()
    print(i)
    candidate_indices = set()
    for elem in target_set:
        candidate_indices.update(element_to_sets.get(elem, []))
    # 计算交集
    intersection_results = [target_set & set_list[i] for i in candidate_indices if target_set & set_list[i]]
    i_sim[i][j] = 1/(len(intersection_results)+alpah)
    i_sim[j][i] = i_sim[i][j]
    print('--train--',elapsed_time)
    pdb.set_trace()



for i in i_ulist: 
    start_time = time.time()
    print(i)
    for j in i_ulist:
       i_sim[i][j] = 1/(len(i_ulist[i]&i_ulist[j])+alpah)
       i_sim[j][i] = 1/(len(i_ulist[i]&i_ulist[j])+alpah)
    elapsed_time = time.time() - start_time
    print('--train--',elapsed_time)

list_user_pair = []
# pos_u_v = np.zeros((len(u_ilist),len(u_ilist))) #reid 之后就可以用了。
pos_u_v = np.zeros((u_id_max+1,u_id_max+1))
for u,v in list_user_pair:
    same_item = u_ilist[u] &u_ilist[v] 
    sim_uv = 0
    for i_one in same_item:
        for j_one in same_item:
            sim_uv+=i_sim[i_one][j_one]
    # u_v_id = u+'-'+v
    pos_u_v[u][v] = sim_uv
    pos_u_v[v][u] = sim_uv

pdb.set_trace()





