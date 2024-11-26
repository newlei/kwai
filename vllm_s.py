import json
import requests
import pdb
# JSON 文件路径
json_file = '../data_process/core'+str(10)+'/train/data_kg_llm.json'
# 服务器地址
server_url = "http://101.6.69.60:5000/process"


# pdb.set_trace()

batch_size = 64
# 读取 JSON 文件并逐条发送请求
def send_requests():
    try:
        # 加载 JSON 文件
        # with open(json_file, 'r') as file:
        #     # data = json.load(file)

        batch_size=0
        batch=[]
        file =open(json_file, 'r') 
        count =0 
        for one_data in file.readlines(): 
            batch.append(one_data)  
            if batch_size<=4:
                batch_size+=1
                continue
            print(f"发送第 {count} 批请求，包含 {len(batch_size)} 条记录")
            
            try:
                response = requests.post(server_url, json=batch)  # 批量发送数据
                if response.status_code == 200:
                    print(f"响应：{response.json()}")
                else:
                    print(f"错误：状态码 {response.status_code}, 内容 {response.text}")
            except requests.RequestException as e:
                print(f"请求失败：{e}")

            exit()


    except FileNotFoundError:
        print(f"文件 {json_file} 未找到")
    except json.JSONDecodeError:
        print(f"文件 {json_file} 不是有效的 JSON 文件")

if __name__ == "__main__":
    send_requests()
