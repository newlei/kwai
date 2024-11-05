import numpy as np 
import pdb 
import time

timestamp1 = 1725638400
timestamp2 = 1725552000
time_local1 = time.localtime(timestamp1)
time_local2 = time.localtime(timestamp2)
print(time_local1,time_local2)

dt1 = time.strftime("%Y-%m-%d %H:%M:%S",time_local1)
dt2 = time.strftime("%Y-%m-%d %H:%M:%S",time_local2)
print(dt1,dt2)

pdb.set_trace()



u_ilist = dict()
i_ulist = dict()


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


            



