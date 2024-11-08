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

file_poi = '../poi_pdate_20241104.csv'
poi_att = pd.read_csv(file_poi, sep='|')
print('poi_att',poi_att.size)
poi_att['poi_id'] = poi_att['poi_id'].fillna(0.0).astype('int', errors='ignore')

# df.astype({'value': 'int'}, errors='ignore')
# poi_att[poi_att[poi_att['poi_id']=='seemly假发'].index]

merged_table = pd.merge(data_interaction, user_att, on=['user_id'], how='inner')
merged_table2 = pd.merge(data_interaction, poi_att, on=['poi_id'], how='inner')

pdb.set_trace()



chunksize = 10 ** 6
file_photo = '../photo_pdate_20241104.csv'
photo_list =[]
for chunk in pd.read_csv(file_photo,  usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], chunksize=chunksize, sep='|', lineterminator='\n'):
    photo_list.append(chunk) 

print(len(photo_list))

