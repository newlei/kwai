import json
import numpy as np 
import pandas as pd 
import pdb
import time
from geopy.distance import geodesic


file_name = '../data_process/core10/data_interaction_final_reid.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_us','ulat','ulong','plat','plong'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')
# pdb.set_trace()

# data_interaction['distance_km'] = data_interaction.apply(lambda row: geodesic((row['ulat'], row['ulong']), (row['plat'], row['plong'])).kilometers, axis=1)

data_interaction['distance_km'] = data_interaction.apply(lambda row: round(geodesic((row['ulat'], row['ulong']), (row['plat'], row['plong'])).kilometers, 2), axis=1)


file_name = '../data_process/core10/data_interaction_final_cat_u_att_reid.csv'
data_interaction_u_att = pd.read_csv(file_name, sep='|')

file_name = '../data_process/core10/data_interaction_final_cat_poi_att_reid.csv'
data_interaction_poi_att = pd.read_csv(file_name, sep='|')

# file_name = '../data_process/core10/data_interaction_final_cat_p_att.csv'
# data_interaction_photo_att = pd.read_csv(file_name, sep='|')

u_att_dict = dict()
#去重复的
data_interaction_u_att = data_interaction_u_att.drop_duplicates(subset='user_id') 
#u_gender|u_age|u_age_part|u_city|play_duration|u_region
for index, row in data_interaction_u_att.iterrows():
    # print(index) # 输出每行的索引值
    user_id = row['user_id']
    if user_id not in u_att_dict:
        u_att_dict[user_id] = "性别是"+str(row['u_gender'])+",年龄是"+str(row['u_age'])+",居住地是"+str(row['u_city'])+",居住地属于中国"+str(row['u_region'])


poi_att_dict = dict()
#去重复的
data_interaction_poi_att = data_interaction_poi_att.drop_duplicates(subset='poi_id') 
#|user_id|photo_id|time_second|poi_id|poi_name|category_name|cate_2_name|cate_1_name|province_name|city_name|brand_name
for index, row in data_interaction_poi_att.iterrows():
    # print(index) # 输出每行的索引值
    poi_id = row['poi_id']
    if poi_id not in poi_att_dict:
        poi_att_dict[poi_id] = "名称是"+str(row['poi_name'])+",类型是"+str(row['category_name'])+"-"+str(row['cate_2_name'])+"-"+str(row['cate_1_name'])+",所在地是"+str(row['city_name'])


data = []

instruction = "针对时空场景的推荐问题，请总结产品在时空场景的推荐偏好包括：时间偏好，空间偏好，时空整体偏好，用户画像，总体偏好，每个偏好用一句话描述，其中总体偏好是结合时间偏好，空间偏好，时空整体偏好和用户画像偏好形成的。此外，对于出现的时间信息需先分类成，早上，上午，中午，下午，傍晚，晚上，凌晨，工作日，节假日，法定节假日等多种细粒度的时间标签，然后用于推理总结偏好" 


# data_interaction = data_interaction.groupby('user_id').agg(list).reset_index()
data_interaction = data_interaction.groupby('poi_id').agg(list).reset_index()
for index, row in data_interaction.iterrows():
    poi_id = row['poi_id'] 
    
    text = ""
    try:
        text = "产品"+"ID是："+str(poi_id)+","+ poi_att_dict[poi_id]#u_att_dict[user_id]
    except:
        text = ""
    text += "\\n 产品被交互的用户序列如下：\\n"
    uid_list = row['user_id']
    time_list = row['time_us']
    distance_list = row['distance_km']
    count_sel = -1
    for user_id in uid_list:  
        count_sel+=1 

        try:
            time_local = time.localtime(time_list[count_sel])  
            dt1 = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            distance_local = distance_list[count_sel]
            text = text+ "在"+dt1+"时间被相对距离为"+str(distance_local)+"KM的用户交互了，用户的ID是："+str(user_id)+","+u_att_dict[user_id]
            text+='\\n'
        except:
            continue
    data.append({
        "instruction": instruction,
        "input": text
    })

output_file = '../data_process/core'+str(10)+'/data_kg_llm_item.json'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')