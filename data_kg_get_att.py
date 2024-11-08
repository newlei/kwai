import numpy as np 
import pdb 
import pandas as pd
import os.path
import sys
import csv
csv.field_size_limit(sys.maxsize)


#交互数据提取：user_id|photo_id|time_second|poi_id
file_name = '../data_process/core10/data_interaction_final.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
print('data_interaction:',data_interaction.shape)



chunksize = 10 ** 6
file_photo = '../photo_pdate_20241104.csv'
photo_list =[]
flag = 0
for chunk in pd.read_csv(file_photo,  usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], chunksize=chunksize, sep='|', lineterminator='\n'):
    photo_list.append(chunk) 
    flag+=1
    if flag ==1:
        data_pid = chunk
    elif flag>1:
        data_pi = pd.concat([data_pid, chunk], axis=0)
    data_pi = data_pi.drop_duplicates(subset='photo_id')

print(len(photo_list))
print(data_pi.shape)

pdb.set_trace()




file_user = '../user_pdate_20241104.csv'
user_att = pd.read_csv(file_user, usecols=['user_id','photo_id','time_second','poi_id','label','play_duration','poi_page_stay_time'], sep='|')
user_att.rename(columns={'user_id': 'user_id', 'photo_id': 'u_gender','time_second': 'u_age','poi_id': 'u_age_part','label': 'u_city','poi_page_stay_time': 'u_region'}, inplace=True)
print('user_att',user_att.shape)

# poi_id|poi_name|category_id|category_name|cate_2_id|cate_2_name|cate_1_id|cate_1_name|country|province_id|province_name|city_id|city_name|district_id|district_name|town_name|brand_name|is_busi_goods|brand_level_reco|collect_poi_user_num

file_poi = '../poi_pdate_20241104.csv'
poi_att = pd.read_csv(file_poi,usecols=['poi_id','poi_name','category_name','cate_2_name','cate_1_name','province_name','city_name','brand_name'], sep='|')
print('poi_att',poi_att.shape)
poi_att1 = poi_att[pd.to_numeric(poi_att['poi_id'], errors='coerce').notnull()]
poi_att1['poi_id'] = poi_att1['poi_id'].astype('int64') 


# user att中有太多重复的了。
# print(user_att['user_id'].duplicated().sum())
user_att_unique = user_att.drop_duplicates(subset='user_id') 
poi_att_unique = poi_att1.drop_duplicates(subset='poi_id') 
data_interaction_u = data_interaction.drop_duplicates(subset='user_id')

# print(poi_att_unique['poi_id'].duplicated().sum())
# merged_table0 = pd.merge(data_interaction_u, poi_att_unique, on=['poi_id'], how='inner') 

# data_interaction.shape  (3201191, 4)
merged_uatt = pd.merge(data_interaction,  user_att_unique,  on=['user_id'], how='inner')
merged_poiatt = pd.merge(data_interaction, poi_att_unique, on=['poi_id'], how='inner')

print('merged_uatt:',merged_uatt.shape)
print('merged_poiatt:',merged_poiatt.shape)

file_name = '../data_process/core'+str(10)+'/data_interaction_final_cat_u_att.csv'
merged_uatt.to_csv(file_name, sep='|')
file_name = '../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att.csv'
merged_poiatt.to_csv(file_name, sep='|')


pdb.set_trace()



