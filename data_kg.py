import numpy as np 
import pdb 
import pandas as pd
import os.path

# import ray
# ray.init(num_cpus=4)
# import modin.pandas as pd


#交互数据提取：user_id|photo_id|time_second|poi_id
#购买行为数据
# data_interaction = pd.read_csv('../photo_payorder_pdate_20241104.csv', usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
#点击行为数据
# data_interaction = pd.read_csv('../goods_click_pdate_20241105.csv', usecols=['user_id','photo_id','poi_id','time_second'], sep='|')


data_interaction = pd.read_csv('../llm_graph_data/user_poi_lat_long_pdate_20241105.csv', usecols=['uid','poi_id','photo_id','time_us','ulat','ulong','plat','plong'], sep='|',engine="c")
print(data_interaction.shape)
data_interaction = data_interaction.rename(columns={"uid": "user_id"})



#poi_id == 0是没有意义的数据，所以直接过滤掉。
data_interaction1 = data_interaction.drop(data_interaction[data_interaction['poi_id']==0].index)
print(data_interaction1.shape)





#先得到了data_interaction8.csv,没有和原始的数据合并。这里重新处理一下，得到最终的清洗结果。
print("merge the data to get final interaction")
file_name = '../data_process/core'+str(10)+'/data_interaction8.csv'
if os.path.isfile(file_name): 
    data_interaction8 = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|') 
    merged_table = pd.merge(data_interaction8, data_interaction1, on=['user_id', 'poi_id'], how='inner')
    print(merged_table.shape)
    pdb.set_trace()

    file_name = '../data_process/core'+str(10)+'/data_interaction_final.csv'
    merged_table.to_csv(file_name, sep='|')
    
    exit()



#先只看主要的交互行为，并统计用户交互行为数量，以及pid被点击的数量。
data_interaction2 =  data_interaction1[['user_id', 'poi_id']]#photo_id
 
def data_process(data_interaction2,core,epoch):
        
    data_interaction2_uid = data_interaction2.groupby('user_id')["poi_id"].apply(list).reset_index(name="poi_id")
    data_interaction2_pid = data_interaction2.groupby('poi_id')["user_id"].apply(list).reset_index(name="user_id")

    #过滤掉，交互行为为core以下的。
    data_interaction2_uid_f1 = data_interaction2_uid[data_interaction2_uid['poi_id'].apply(lambda x: len(x) > core)]
    data_interaction2_pid_f1 = data_interaction2_pid[data_interaction2_pid['user_id'].apply(lambda x: len(x) > core)]
    print(data_interaction2_uid_f1.shape,'user >core:',core)
    print(data_interaction2_pid_f1.shape,'item >core:',core)
    # pdb.set_trace()

    #拆分，过滤后的数据，然后进行比对，保留2个表中u_id和p_id都出现数据，用于下一轮的过滤
    data_interaction2_uid_f1_split =  pd.DataFrame([
        [u, p] for u, P in data_interaction2_uid_f1.itertuples(index=False)
        for p in P 
    ], columns=data_interaction2_uid_f1.columns)
    data_interaction2_pid_f1_split =  pd.DataFrame([
        [u, p] for u, P in data_interaction2_pid_f1.itertuples(index=False)
        for p in P 
    ], columns=data_interaction2_pid_f1.columns)

    # print(data_interaction2_uid_f1_split.shape)
    # print(data_interaction2_pid_f1_split.shape)

    #交换2列的顺序。
    data_interaction2_pid_f1_split[['user_id', 'poi_id']] = data_interaction2_pid_f1_split[['poi_id', 'user_id']]
    #拼接，并保留重复的，就是2个表中均出现的数据，保留下来的就是，共同的部分。即user_id和photo_id，均出现在2张表（data_interaction2_uid_f1_split, data_interaction2_pid_f1_split）
    data_interaction3_cat = pd.concat([data_interaction2_uid_f1_split, data_interaction2_pid_f1_split], axis=0)
    #  使用duplicated()方法查找重复行
    duplicates3 = data_interaction3_cat.duplicated()
    #  使用布尔索引选择重复行,这样就保留了重复的行
    duplicate3_all = data_interaction3_cat[duplicates3]
    # 对于重复的行，只保留一个数据就行了。
    duplicate3 = duplicate3_all.drop_duplicates()
    
    str_out = "使用"+str(core)+"-core,第"+str(epoch)+"轮清洗之后的行为数量:"
    print(str_out,duplicate3.shape) #357366 >2  462012 >1
    return duplicate3

core_num = 5
print('core_num:',core_num)
data_interaction3 = data_process(data_interaction2,core=core_num,epoch=1)
data_interaction4 = data_process(data_interaction3,core=core_num,epoch=2)
data_interaction5 = data_process(data_interaction4,core=core_num,epoch=3)
data_interaction6 = data_process(data_interaction5,core=core_num,epoch=4)
data_interaction7 = data_process(data_interaction6,core=core_num,epoch=5)
data_interaction8 = data_process(data_interaction7,core=core_num,epoch=6)

file_name = '../data_process/core'+str(core_num)+'/data_interaction3.csv'
data_interaction3.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(core_num)+'/data_interaction4.csv'
data_interaction4.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(core_num)+'/data_interaction5.csv'
data_interaction5.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(core_num)+'/data_interaction6.csv'
data_interaction6.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(core_num)+'/data_interaction7.csv'
data_interaction7.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(core_num)+'/data_interaction8.csv'
data_interaction8.to_csv(file_name, sep='|')


#获得清洗后的数据
merged_table = pd.merge(data_interaction8, data_interaction1, on=['user_id', 'poi_id'], how='inner')
file_name = '../data_process/core'+str(core_num)+'/data_interaction_final.csv'
merged_table.to_csv(file_name, sep='|')

# goods_click_pdate_20241105
# 使用6-core,第1轮清洗之后的行为数量: 83803880
# 使用6-core,第2轮清洗之后的行为数量: 45008556
# 使用6-core,第3轮清洗之后的行为数量: 17144888 
# goods_click_pdate_20241105
# 用10-core,第1轮清洗之后的行为数量: 74768032
# 使用10-core,第2轮清洗之后的行为数量: 29794666
# 使用10-core,第3轮清洗之后的行为数量: 5685002
# 使用10-core,第4轮清洗之后的行为数量: 4904226
# 使用10-core,第5轮清洗之后的行为数量: 4525662
# 使用10-core,第6轮清洗之后的行为数量: 4396544

#最新的数据user_poi_lat_long_pdate_20241105

# pdb.set_trace()
print("merge the data to get final interaction")
data_interaction8 = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|') 
merged_table = pd.merge(data_interaction8, data_interaction1, on=['user_id', 'poi_id'], how='inner')

file_name = '../data_process/core'+str(core_num)+'/data_interaction_final.csv'
merged_table.to_csv(file_name, sep='|')


exit()










pdb.set_trace()



count = 0
user_pid = dict()
user_poi = dict()
for index, row in data_interaction.iterrows():
    # print(index) # 输出每行的索引值
    if row['poi_id']==0:
        data_interaction.drop(index=index)
    count+=1

print(data_interaction.shape)
print(count)

pdb.set_trace()


exit()


# data_path='../'
# data_file = pd.read_csv(data_path, sep='|')

df1 = pd.read_csv('../poi_pdate_20241104.csv', usecols=['poi_id'], sep='|') # 按列名，列名必须存在  
df2 = pd.read_csv('../photo_pdate_20241104.csv', usecols=['poi_id'], sep='|', lineterminator='\n') # 按列名，列名必须存在 


df3 = pd.concat([df1, df2]).drop_duplicates(keep=False)
#  df1.append(df2).drop_duplicates(keep=False)

#476890, 224728631 , 1945239
print(df3.shape,df2.szie,df1.shape)


pdb.set_trace()


data_poi_path='../poi_pdate_20241104.csv'
poi_data = pd.read_csv(data_poi_path, usecols=['poi_id', 'poi_name','category_id','category_name','cate_2_id','cate_2_name','cate_1_id','cate_1_name'], sep='|') # 按列名，列名必须存在  

poi_id = set()
for index, row in poi_data.iterrows():
    # print(index) # 输出每行的索引值
    poi_id.add(row['poi_id'])
    # pdb.set_trace()

data_photo_path='../photo_pdate_20241104.csv'
photo_data = pd.read_csv(data_photo_path, usecols=['photo_id', 'poi_id','poi_name','origin_poi_id','photo_cate_type','photo_second_cate_type'], sep='|', lineterminator='\n') # 按列名，列名必须存在  

photo_poi_id = set()
for index, row in photo_data.iterrows():
    # print(index) # 输出每行的索引值
    photo_poi_id.add(row['poi_id'])
    # pdb.set_trace()

p_count = len(photo_poi_id-poi_id)/len(photo_poi_id)

pdb.set_trace()


count_all=0
count = 0
for index, row in photo_data.iterrows():
    # print(index) # 输出每行的索引值
    count_all+=1
    if row['poi_id'] in poi_id:
        count+=1

print(count/count_all)

pdb.set_trace()







exit()


for index, row in data_file.iterrows():
    print(index) # 输出每行的索引值

data_good_path='../goods.csv'
pd.read_csv(data_good_path, usecols=['goods_id', 'goods_name','goods_cate_1st_name','goods_cate_2nd_name','goods_cate_3rd_name','brand_name']) # 按列名，列名必须存在  



data_good_path='../interaction.csv'
pd.read_csv(data_good_path, usecols=['goods_id', 'goods_name','goods_cate_1st_name','goods_cate_2nd_name','goods_cate_3rd_name','brand_name']) # 按列名，列名必须存在  
