import json
import requests

# JSON 文件路径
json_file = '../data_process/core'+str(10)+'/train/data_kg_llm.json'
# 服务器地址
server_url = "http://101.6.69.60:5000/process"

pdb.set_trace()

batch_size = 64
# 读取 JSON 文件并逐条发送请求
def send_requests():
    try:
        # 加载 JSON 文件
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # 确保是列表格式
        if not isinstance(data, list):
            print("JSON 文件内容不是列表格式")
            return

        # 分批发送数据
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]  # 获取当前批次数据
            print(f"发送第 {i//batch_size + 1} 批请求，包含 {len(batch)} 条记录")

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
