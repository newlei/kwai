import numpy as np 
import pdb 
import time
import pandas as pd

file_name = '../data_process/core10/data_interaction_final.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')

u_ilist = dict()
i_ulist = dict()

data_interaction = data_interaction.groupby('user_id').agg(list).reset_index()
for index, row in data_interaction.iterrows():
    user_id = row['user_id']
    poi_list = row['poi_id']
    if user_id not in u_ilist:
        u_ilist[user_id]=set(poi_list)
    pdb.set_trace()



u_ilist[u] = set()
i_ulist[i] = set()

#针对任意item i和j，就计算交集，计算得到值，构建成矩阵。
alpah=0.1
i_sim = np.zeros((len(i_ulist),len(i_ulist)))
for i in i_ulist:
    for j in i_ulist:
       i_sim[i][j] = 1/(i_ulist[i]&i_ulist[j]+alpah)

for u,v in list_user_pair:
    same_item = u_ilist[u] &u_ilist[v] 
    sim_uv = 0
    for i_one in same_item:
        for j_one in same_item:
            sim_uv+=i_sim[i_one][j_one]


            



