from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
from vllm import LLM, SamplingParams



# Step 1: 初始化模型
# model_path ="Qwen/Qwen2.5-1.5B-Instruct"  
model_path ="Qwen/Qwen2.5-3B-Instruct"
llm = LLM(model=model_path, dtype='half', tensor_parallel_size=2, max_model_len=128000) 


# Step 2: 定义批量输入数据
batch_data = [
    "请简述量子计算的基本原理。",
    "给出关于人工智能伦理的三条建议。",
    "如何评价最近的大模型技术发展？",
]

# Step 3: 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=1500,  # 限制生成的最大长度
)

# # Step 4: 执行批量推理
# results = llm.generate(batch_data, sampling_params)

# # Step 5: 输出结果
# for i, output in enumerate(results):
#     print(f"输入: {batch_data[i]}")
#     print(f"生成结果: {output.outputs[0].text}\n")

 

count=0
input_texts =[] 
# json_path = '../data_process/core'+str(10)+'/train/data_kg_llm.json'
json_path = '../data_process/core'+str(10)+'/train/data_kg_llm_item.json'
elapsed_time_all = 0
elapsed_time_count = 0

res_data = []
with open(json_path, 'r', encoding="utf-8") as f:
    # 读取所有行 每行会是一个字符串
    batch_size=0
    batch_data=[]
    batch_data_id = []
    start_time = time.time()
    for one_data in f.readlines(): 
        count+=1
        # 将josn字符串转化为dict字典
        prompt_one = json.loads(one_data)  
        # str_in = prompt_one["data"]["instruction"]+"上下文信息："+prompt_one["data"]["input"]+"\n \n 写出总结性的回答，不包含原句重复。"
        # batch_data.append(str_in)

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(prompt_one["data"])}
            ]
        batch_data.append(messages) 
        # batch_data_id.append(prompt_one["user_id"])
        batch_data_id.append(prompt_one["poi_id"])

        if batch_size<=1022:
            batch_size+=1
            continue
 
        response = llm.chat(batch_data, sampling_params)
 
        for i, output in enumerate(response): 
            # res_data.append({
            #     "user_id":  batch_data_id[i],
            #     "data": output.outputs[0].text
            # }) 
            res_data.append({
                "poi_id":  batch_data_id[i],
                "data": output.outputs[0].text
            }) 


        elapsed_time = time.time() - start_time
        elapsed_time_all+=elapsed_time
        elapsed_time_count+=1
        elapsed_time_average = elapsed_time_all/elapsed_time_count
        print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average,'---count---',count)
        # pdb.set_trace()
        batch_data = []
        batch_data_id = []
        batch_size = 0
        start_time = time.time()
        # if count>2:
        #     break
        # # pdb.set_trace()
        # count+=1

    if  batch_size>0:
        print("laster batch", batch_size)
        response = llm.chat(batch_data, sampling_params)
        for i, output in enumerate(response): 
            # res_data.append({
            #     "user_id":  batch_data_id[i],
            #     "data": output.outputs[0].text
            # }) 
            res_data.append({
                "poi_id":  batch_data_id[i],
                "data": output.outputs[0].text
            }) 


# json_res_path = '../data_process/core'+str(10)+'/train/data_kg_llm_summary.json'
json_res_path = '../data_process/core'+str(10)+'/train/data_kg_llm_summary_item.json'
with open(json_res_path, 'w', encoding='utf-8') as f:
    for item in res_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

f1 = open(json_res_path, 'r', encoding="utf-8")
for one_data in f1.readlines(): 
    prompt_one = json.loads(one_data) 


# response1 = llm.chat(batch_data, sampling_params)
# res_data1= res_data
# for i, output in enumerate(response1):   res_data1.append({"user_id":  batch_data_id[i],"data": output.outputs[0].text}) 
# json_res_path = '../data_process/core'+str(10)+'/train/data_kg_llm_summary1.json'
# f= open(json_res_path, 'w', encoding='utf-8') 
# f.write('\n'.join(map(lambda item: json.dumps(item, ensure_ascii=False), res_data1)) + '\n')


print('end')
pdb.set_trace()
exit()
