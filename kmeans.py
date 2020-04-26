from nilmtk import DataSet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.cluster import KMeans
import multiprocessing


def train_cluster(train_vecs, model_name=None, start_k=2, end_k=10):
    print('training cluster')
    SSE = []
    SSE_d1 = []  # sse的一阶导数
    SSE_d2 = []  # sse的二阶导数
    models = []  # 保存每次的模型
    for i in range(start_k, end_k):
        kmeans_model = KMeans(n_clusters=i, n_jobs=multiprocessing.cpu_count(), )
        kmeans_model.fit(train_vecs)
        SSE.append(kmeans_model.inertia_)  # 保存每一个k值的SSE值
        print('{} Means SSE loss = {}'.format(i, kmeans_model.inertia_))
        models.append(kmeans_model)
    SSE_length = len(SSE)
    for i in range(1, SSE_length):
        SSE_d1.append((SSE[i - 1] - SSE[i]) / 2)
    for i in range(1, len(SSE_d1) - 1):
        SSE_d2.append((SSE_d1[i - 1] - SSE_d1[i]) / 2)

    best_model = models[SSE_d2.index(max(SSE_d2)) + 1]

    # getGausian(train_vecs,best_model)
    return best_model




def adaptive_kmeans(house=1):
    path = 'data/low_freq/house_%s/' % house
    df = pd.read_csv(path+ 'labels.dat',header=-1,sep=' ')

    print(df[1].values)

    for i in range(2,len(df)):
        dev_name = df[1][i]
        print(dev_name)

        data =pd.read_csv(path+ 'channel_%d.dat'%i,header=-1,sep=' ')
        data.index = data[0]
        del data[0]

        kmodel = train_cluster(data[1].values.reshape(-1, 1))
        data.columns = ['value']
        data['label'] = kmodel.labels_

        data.to_csv('house_'+house+'/'+dev_name+str(i)+'.csv')

if __name__ == '__main__':
    # 表示对house1的数据进行有限状态机建模
    adaptive_kmeans(house=1)
