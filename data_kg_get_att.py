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
print(data_interaction.size)



file_user = '../user_pdate_20241104.csv'
user_att = pd.read_csv(file_user, usecols=['user_id','photo_id','time_second','poi_id','label','play_duration','poi_page_stay_time'], sep='|')
user_att.rename(columns={'user_id': 'user_id', 'photo_id': 'u_gender','time_second': 'u_age','poi_id': 'u_age_part','label': 'u_city','poi_page_stay_time': 'u_region'}, inplace=True)
print('user_att',user_att.size)

# file_photo = '../photo_pdate_20241104.csv'
# photo_att = pd.read_csv(file_photo, usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], sep='|', lineterminator='\n')

# poi_id|poi_name|category_id|category_name|cate_2_id|cate_2_name|cate_1_id|cate_1_name|country|province_id|province_name|city_id|city_name|district_id|district_name|town_name|brand_name|is_busi_goods|brand_level_reco|collect_poi_user_num

file_poi = '../poi_pdate_20241104.csv'
poi_att = pd.read_csv(file_poi,usecols=['poi_id','poi_name','category_name','cate_2_name','cate_1_name','province_name','city_name','brand_name'], sep='|')
print('poi_att',poi_att.size)
poi_att1 = poi_att[pd.to_numeric(poi_att['poi_id'], errors='coerce').notnull()]
poi_att2 = poi_att1['poi_id'].astype('int64') 

merged_table = pd.merge(data_interaction, user_att, on=['user_id'], how='inner')
merged_table2 = pd.merge(data_interaction, poi_att2, on=['poi_id'], how='inner')

print('merged_table:',merged_table.size)
print('merged_table2:',merged_table2.size)

pdb.set_trace()



chunksize = 10 ** 6
file_photo = '../photo_pdate_20241104.csv'
photo_list =[]
for chunk in pd.read_csv(file_photo,  usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], chunksize=chunksize, sep='|', lineterminator='\n'):
    photo_list.append(chunk) 

print(len(photo_list))

