# -- coding: utf-8 --
import json
import requests
import pdb
import numpy as np
import time
import random

# JSON 文件路径
# json_file = '../data_process/core'+str(10)+'/train/data_kg_llm.json'
json_file = '../data_process/core'+str(10)+'/train/data_kg_llm_item.json'
# 服务器地址
server_url = "http://101.6.69.60:8001/process"


# pdb.set_trace()

batch_size = 1024
# 读取 JSON 文件并逐条发送请求
def send_requests():
    try:
        # 加载 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as file:
            buffer = []
            for line in file:
                try:
                    # 解析每一行 JSON
                    data = json.loads(line)
                    buffer.append(data)
                    # 当缓冲区达到 batch_size 时，发送请求
                    if len(buffer) == batch_size:
                        response = requests.post(server_url, json=buffer)
                        print(f"Sent {batch_size} items. Response: {response.status_code}, {response.json()}")
                        buffer = []  # 清空缓冲区
                        s_time= int(10*random.random())+1
                        time.sleep(s_time)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:20]}{line[-20:]}. Error: {e}")
                
            
            # 发送剩余数据
            if buffer:
                response = requests.post(server_url, json=buffer)
                print(f"Sent remaining {len(buffer)} items. Response: {response.status_code}, {response.json()}")



    #    #发生单行数据
    #     with open(json_file, 'r', encoding='utf-8') as f:
    #         for i, line in enumerate(f):
    #             try:
    #                 # 解析 JSON 数据
    #                 json_data = json.loads(line.strip())
                    
    #                 # 发送 POST 请求
    #                 response = requests.post(server_url, json=json_data)
                    
    #                 # 打印结果
    #                 print(f"Line {i + 1}: Status {response.status_code}, Response: {response.json()}")
    #             except json.JSONDecodeError:
    #                 print(f"Line {i + 1}: Invalid JSON format")
    #             except requests.RequestException as e:
    #                 print(f"Line {i + 1}: Request failed with error: {e}")


    except FileNotFoundError:
        print(f"文件 {json_file} 未找到")
    except json.JSONDecodeError:
        print(f"文件 {json_file} 不是有效的 JSON 文件")

if __name__ == "__main__":
    send_requests()
