from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-2.5-1.5B")


def llm_summary(batch_data):
    # Tokenize batch data
    inputs = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取生成的结果
    generated_ids = outputs.logits.argmax(dim=-1)
    results = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    for i, text in enumerate(batch_data):
        print(f"输入: {text}")
        print(f"生成结果: {results[i]}")
        print("-" * 30)
    return results



count=0
input_texts =[] 
json_path = '../data_process/core'+str(10)+'/data_kg_llm.json'
elapsed_time_all = 0
elapsed_time_count = 0

res_data = []
with open(json_path, 'r', encoding="utf-8") as f:
    # 读取所有行 每行会是一个字符串
    batch_count =0 
    batch_data =[]
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        start_time = time.time()
        prompt_one = json.loads(one_data)  
        batch_data.append(str(prompt_one["data"]))
        if batch_count<32:
            batch_count+=1
            continue

        response_one = llm_summary(batch_data)
        input_texts.append(response_one)
        # prompt_one = json.loads(one_data)
        # llm_summary(prompt_one)
        print("user_id",prompt_one["user_id"])
        res_data.append({
            "user_id":  prompt_one["user_id"],
            "data": response_one
        }) 
        elapsed_time = time.time() - start_time
        elapsed_time_all+=elapsed_time
        elapsed_time_count+=1
        elapsed_time_average = elapsed_time_all/elapsed_time_count
        print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average)
        batch_data = []
        # if count>4:
        #     break
        # pdb.set_trace()
        # count+=1
