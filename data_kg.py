import numpy as np 
import pdb 
import pandas as pd


#交互数据提取：user_id|photo_id|time_second|poi_id
data_interaction = pd.read_csv('../photo_payorder_pdate_20241104.csv', usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
print(data_interaction.size)

data_interaction1 = data_interaction.drop(data_interaction[data_interaction['poi_id']==0].index)
print(data_interaction1.size)

data_interaction2 = data_interaction1.groupby('user_id').apply(lambda x: x[['photo_id', 'poi_id', 'time_second']].to_string(index=False)).reset_index(name='item_id')
print(data_interaction2.size)

pdb.set_trace()



count = 0
user_pid = dict()
user_poi = dict()
for index, row in data_interaction.iterrows():
    # print(index) # 输出每行的索引值
    if row['poi_id']==0:
        data_interaction.drop(index=index)
    count+=1

print(data_interaction.size)
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
print(df3.size,df2.szie,df1.size)


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
