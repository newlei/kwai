import json
import numpy as np 
import pandas as pd 
import pdb
import time
from geopy.distance import geodesic


file_name = '../data_process/core10/data_interaction_final_reid.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_us','ulat','ulong','plat','plong'], sep='|')
# data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')
pdb.set_trace()

data_interaction1 = data_interaction1[pd.to_numeric(data_interaction1['user_id'], errors='coerce').notnull()]
data_interaction1['user_id'] = data_interaction1['user_id'].astype('float') 


data_interaction['distance_km'] = data_interaction.apply(lambda row: geodesic((row['ulat'], row['ulong']), (row['plat'], row['plong'])).kilometers, axis=1)


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
# instruction = "针对时空场景的推荐问题，请总结出用户在时空场景的推荐偏好包括：时间偏好，空间偏好，时空整体偏好，产品类型偏好，总体偏好，每个偏好用一句话描述。其中对于出现的时间信息需先分类成，早上，上午，中午，下午，傍晚，晚上，凌晨，工作日，节假日，法定节假日等多种细粒度的时间标签，然后用于推理总结偏好" 
# 根据用户的交互行为和提供的详细时间信息，可以总结出以下用户的时空场景推荐偏好：
# 1. **时间偏好**：用户更倾向于在工作日期间进行餐饮消费，尤其是在周末及节假日前后的几天内。
# 2. **空间偏好**：用户主要在长春市内的多个地点选择餐饮服务，且偏好集中在特定区域如长春大街、太和东街等
# 3. **时空整体偏好**：用户偏好的时段多集中在晚上，特别是晚餐时间（晚6点到9点之间），并且喜欢在节假日或周末前后进行聚餐活动。
# 4. **产品类型偏好**：用户对各类美食都有兴趣，但特别偏好火锅类（如鱼火锅、其他火锅）、烧烤、铁锅炖以及各种小吃快餐。此外，也显示出对地方特色菜肴（如东北菜、新疆菜）的兴趣。
# 5. **总体偏好**：用户倾向于在熟悉的餐馆就餐，并多次重复访问同一家店铺，显示出较高的忠诚度；同时，也愿意尝试不同的美食类型和服务场所。
# 根据用户的交互记录和时间标签分类，可以总结出以下偏好：
# 1. **时间偏好**：用户主要在2024年10月期间进行多次服务交互，且集中在10月7日至14日期间，倾向于选择凌晨时间进行相关服务。
# 2. **空间偏好**：用户频繁访问位于鞍山市的服务地点，显示出对鞍山地区的偏好。
# 3. **时空整体偏好**：用户偏好在特定时间段（如10月份、凌晨）于固定城市（鞍山）内进行服务消费。
# 4. **产品类型偏好**：用户偏好与美容美体相关的服务项目，尤其是涉及皮肤管理和瘦身纤体的类别。
# 5. **总体偏好**：用户倾向于在凌晨时分，在其居住地鞍山，选择美容美体类服务，特别是皮肤管理和瘦身项目。

instruction = "针对时空场景的推荐问题，请总结出用户在时空场景的推荐偏好包括：时间偏好，空间偏好，时空整体偏好，产品类型偏好，总体偏好，每个偏好用一句话描述，其中总体偏好是结合时间偏好，空间偏好，时空整体偏好和产品类型偏好形成的。此外，对于出现的时间信息需先分类成，早上，上午，中午，下午，傍晚，晚上，凌晨，工作日，节假日，法定节假日等多种细粒度的时间标签，然后用于推理总结偏好" 

data_interaction = data_interaction.groupby('user_id').agg(list).reset_index()
for index, row in data_interaction.iterrows():
    user_id = row['user_id']
    text = ""
    try:
        text = "用户"+"ID是："+str(user_id)+","+u_att_dict[user_id]
    except:
        text = ""
    text += "\\n 用户交互的产品序列如下：\\n"
    poi_list = row['poi_id']
    time_list = row['time_us']
    distance_list = row['distance_km']
    count_sel = -1
    for poi_id in poi_list:  
        count_sel+=1 

        try:
            time_local = time.localtime(time_list[count_sel])  
            dt1 = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            distance_local = distance_list[count_sel]
            text = text+ "在"+dt1+"时间且相对距离为"+str(distance_local)+"交互的产品ID是："+str(poi_id)+","+poi_att_dict[poi_id]
            text+='\\n'
        except:
            continue
    data.append({
        "instruction": instruction,
        "input": text
    })

output_file = '../data_process/core'+str(10)+'/data_kg_llm.json'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')