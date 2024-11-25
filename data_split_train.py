import json
import numpy as np 
import pandas as pd 
import pdb
import time
from geopy.distance import geodesic


file_name = '../data_process/core10/data_interaction_final_reid.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_us','ulat','ulong','plat','plong'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 存储分割结果
train_data = []
val_data = []
test_data = []


# 按 user_id 分组处理
for user_id, group in data_interaction.groupby("user_id"):
    # 获取该用户所有的 item_id 索引
    item_indices = group.index
    item_count = len(item_indices)
    
    # 如果用户的交互数小于3，无法严格按比例拆分，直接跳过或全分入训练集
    if item_count < 3:
        train_data.append(group)
        continue
    
    # 生成打乱的索引序列
    shuffled_indices = np.random.permutation(item_indices)
    
    # 按比例切分
    train_end = int(0.7 * item_count)
    val_end = train_end + int(0.1 * item_count)
    
    train_idx = shuffled_indices[:train_end]
    val_idx = shuffled_indices[train_end:val_end]
    test_idx = shuffled_indices[val_end:]
    
    # 将拆分后的数据分别添加到对应列表中
    train_data.append(group.loc[train_idx])
    val_data.append(group.loc[val_idx])
    test_data.append(group.loc[test_idx])

# 合并所有分组的结果
train_df = pd.concat(train_data)
val_df = pd.concat(val_data)
test_df = pd.concat(test_data)

print('data_interaction',data_interaction.shape)
print('train_df',train_df.shape)
print('val_df',val_df.shape)
print('test_df',test_df.shape)

# 保存到 CSV 文件
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)


