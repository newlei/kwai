import csv
import json
import requests
import random
import time

def send_data_in_batches(csv_file, url, batch_size=64):
    """
    读取 CSV 文件并每 batch_size 行发送一次请求到服务器。
    
    Args:
        csv_file (str): CSV 文件路径。
        url (str): 服务器 URL。
        batch_size (int): 每批发送的行数。
    """
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        batch = []
        for row in csv_reader:
            batch.append(row)
            if len(batch) == batch_size:
                send_batch(batch, url)
                batch = []  # 清空当前批次
                s_time= int(5*random.random())+1
                time.sleep(s_time)

        # 发送剩余的行
        if batch:
            send_batch(batch, url)

def send_batch(batch, url):
    """
    将一个批次的行发送到服务器。
    
    Args:
        batch (list): 要发送的数据列表。
        url (str): 服务器 URL。
    """
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"data": batch})
        )
        print(f"Batch sent. Response: {response.status_code}, {response.json()}")
    except Exception as e:
        print(f"Error sending batch: {e}")

if __name__ == "__main__":
    # 服务器 URL
    server_url = "http://101.6.69.60:5000/process"

    # CSV 文件路径
    csv_file_path = '../data_process/core'+str(10)+'/test.csv'  # 替换为实际路径

    # 每 10000 行发送一次
    send_data_in_batches(csv_file_path, server_url, batch_size=10000)
