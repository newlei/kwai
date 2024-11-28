# kwai

### data_kg.py

是用于清洗点击的行为数据，最后生成的数据在

'../data_process/core'+str(10)+'/data_interaction_final.csv

新使用文件是:user_poi_lat_long_pdate_20241105


### data_kg_get_att.py

是用于获得user_id ,photo_id,|poi_id,对应的属性信息。
代码的最后部分是对id进行reid。

最后生成的数据在

- 拼接了user的属性，少了3K个user，交互从300w到了100w
../data_process/core'+str(10)+'/data_interaction_final_cat_u_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_u_att_reid.csv

- 拼接了poi的属性
../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att_reid.csv

- 拼接了photo的属性
../data_process/core'+str(10)+'/data_interaction_final_cat_p_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_p_att_reid.csv

- id的映射关系已保存至
file = open('../data_process/core10/mapping_dict.pkl','rb')

### data_split_train.py

- 输入reid的交互行为：file_name = '../data_process/core10/data_interaction_final_reid.csv'
- 根据user对交互行为进行拆分，train：val：text=0.7:0.1:0.2
- 输出：../data_process/core10/train.csv or val.csv or test,csv


###  data_kg_json_llm.py and data_kg_json_llm_item.py
将生成的数据构建成LLM读取的json文件。文件路径

使用data_kg_json_llm.py ：
- 针对user的数据构建结果： '../data_process/core'+str(10)+'/train/data_kg_llm.json'

使用data_kg_json_llm_item.py ：
- 针对item的数据构建结果：'../data_process/core'+str(10)+'/train/data_kg_llm_item.json'



###  llm_summary.py and llm_summary_small.py
总结用户的时空偏好并提取emb，目前是一条条的送进去，还没有形成batch，因为显存不够。

llm_summary_small.py 改成了，vllm+"Qwen/Qwen2.5-3B-Instruct"，这样batch可以设置为64，1张卡就行了。


输入文件：
- 用户侧的json文件：json_path = '../data_process/core'+str(10)+'/train/data_kg_llm.json'
- 产品侧的json文件：json_path = '../data_process/core'+str(10)+'/train/data_kg_llm_item.json'


输出2个文件：
- 用户侧总结的结果：json_res_path = '../data_process/core'+str(10)+'/train/data_kg_llm_summary.json'
- 产品侧总结的结果：json_res_path = '../data_process/core'+str(10)+'/train/data_kg_llm_summary_item.json'


###  llm_summary_small_emb.py
是在llm_summary_small.py得到结果后，输出user emb和item emb。
- np.save(user_emb,'../data_process/core'+str(10)+'/train/llm_user_emb.pkl')
- np.save(user_emb,'../data_process/core'+str(10)+'/train/llm_item_emb.pkl')


###  data_pos_behavoir.py
用于找到正样本，便于利用对比学习，将信号对齐。
目前1/(len(i_ulist[i]&i_ulist[j])+alpah)计算过程特别慢，因为要计算所有的i，j的pair情况
- 使用稀疏矩阵来计算就很快了。15s就行了
- user set 和 user set之间的交集，也是先用稀疏矩阵来计算，获得有交集的u v id的pair，然后去计算，常规方法计算预计22 min。采用多线程的方法。550s=9min，少了一点时间，还可以接受。在新的数据集上需要3个小时。


###  adapter_network.py
adapter网络，因为现在还没有真实数据，给了一下随机的数据进行测试。目前loss稳定下降。
本来只有一个net，现在又搞了一个decoder，用于loss_reconstruction，就成为了一个Unet，感觉不这样的话，只做对齐协同信号，但是维度降低很多，也需要reconstruction来保留语义信息。

