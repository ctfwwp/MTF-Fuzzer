import time
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import torch

def cosine_distance(a,b):
    dot_pro = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1-(dot_pro/(norm_a * norm_b))
def cluster(handle_data, start_index):
    centroids = np.load("data.npy")
    print(start_index)
    if start_index + 32 < len(handle_data):
        batch = handle_data[start_index:start_index + 32]
        print(batch)
        batch_dis = []
        for han_data in batch:
            dis = []
            for i, cen in enumerate(centroids):
                dis.append(cosine_distance(han_data, cen))
            batch_dis.append(dis)

        start_index += 32
        return torch.tensor(batch_dis), start_index

    batch_dis = []
    for han_data in handle_data[start_index:]:
        dis = []
        for cen in centroids:
            dis.append(cosine_distance(han_data, cen))
        batch_dis.append(dis)
    return torch.tensor(batch_dis), start_index
def data_handle(train_data, max_len):  #数据预处理
    resulet_list = []
    for tra_data in train_data:
        tra_list = []
        data_len = len(tra_data)
        Tab = max_len - data_len
        for tra_str in tra_data:
            if tra_str =='.':
                tra_list.append(11)
            elif tra_str =='-':
                tra_list.append(12)
            elif tra_str =='+':
                tra_list.append(13)
            elif tra_str =='e':
                tra_list.append(14)
            else:
                tra_list.append(int(tra_str))
        for _ in range(Tab):
            tra_list.append(15)
        resulet_list.append(tra_list)
    return np.array(resulet_list)
def data_set_handle():
    dev_data = []
    dev_set = []
    dataset_file = ["dataset/dev_0.txt", "dataset/dev_1.txt", "dataset/dev_2.txt", "dataset/dev_3.txt",
                    "dataset/dev_4.txt", "dataset/dev_5.txt", "dataset/dev_6.txt", "dataset/dev_7.txt"]
    # dataset_file = ["dataset/dev_6.txt"]
    for file_name in dataset_file:
        data = open(file_name, 'r').readlines()
        dev_data = dev_data + data
    for str2 in dev_data:
        dev_set.append(str2.split(' ')[1].replace("\n", ""))

    handle_data = data_handle(dev_set, 63) #处理过后的数据集
    return handle_data

if __name__ == '__main__':
    handle_data = data_set_handle()
    start_index = 0
    for x in range(100):
        outputs_k, n = cluster(handle_data, start_index)  # 用聚类来求结果
        outputs_k = F.softmax(outputs, dim=1)  # 用softmax做归一化
        start_index = n
