from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

from keras.models import Model

from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout, Dense

from sklearn.cluster import KMeans
from sklearn.cluster import Birch

from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import metrics

import numpy as np
import fairies as fa
from tqdm import tqdm
import random
import copy
import joblib
import json

from utils import *
import os

def get_vec(
        datas,         
        save_path,
        model_config_path,
        model_checkpoint_path,
        model_dict_path,
        maxlen = 64,
        vector_name = 'cls',
        model_type = 'sentence_pair',
        isElectra = True 
    ):

    """
        datas 需要转换特征的数据。
              如果使用句对模型数据格式应该是
              [
                (文本1, 文本2, 标签id),
                (文本1, 文本2, 标签id),
                ...
              ]
              如果使用的是单句模型格式应该是
              [
                (文本, 标签id),
                (文本, 标签id),
                ...
              ]

        save_path 特征文件保存路径,保存为.npy文件
    """

    # 词向量获取方法 cls,mean,
    tokenizer = Tokenizer(model_dict_path, do_lower_case=True)  # 建立分词器

    if isElectra:
        model = build_transformer_model(model_config_path, model_checkpoint_path,model='electra')  
    else:
        model = build_transformer_model(model_config_path, model_checkpoint_path)

    output = []
    print('开始提取')
    for r in tqdm(datas):
        if model_type == "sentence_pair":
            token_ids, segment_ids = tokenizer.encode(r[0],r[1],maxlen=maxlen)
        else:
            token_ids, segment_ids = tokenizer.encode(r[0],maxlen=maxlen)
             
        if vector_name == 'cls':
            cls_vector = model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
            output.append(cls_vector)
        elif vector_name == 'mean':
            new = []
            vector = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
            for i in range(768):
                temp = 0
                for j in range(len(vector)):
                    temp += vector[j][i]
                new.append(temp/(len(vector)))            
            output.append(new)

    print('保存数据')
    fa.write_npy(save_path,output)

def kMeans(
    vecs_path,
    datas_path,
    n_clusters,
    save_path
):
    feature = fa.read(vecs_path)

    clf = KMeans(n_clusters=n_clusters)
    s = clf.fit(feature)
    joblib.dump(clf, 'model/cluster_{}.pkl'.format(n_clusters))
    pre = clf.predict(feature)
    data = fa.read_json(datas_path)
    
    new = []
    
    for i,d in enumerate(data):
        d['cluster_label'] = int(pre[i])
        new.append(d)

    fa.write_json(save_path,new,isIndent = True)

def split_train_data_by_cluster(
        datas,
        label_name,
        cluster_name,
        test_size = 0.1
    ):

    data_dicts = {}

    train_data = []
    test_data = []

    for data in datas:
        
        classify_name = str(data[label_name]) + '_' + str(data[cluster_name])
        if classify_name not in data_dicts:
            data_dicts[classify_name] = []
        data_dicts[classify_name].append(data)
            
    for classify_name in data_dicts:
        
        num = len(data_dicts[classify_name])
        if num == 0:
            continue
        
        test_num = int(num*test_size)
        random.shuffle(data_dicts[classify_name])

        test_data.extend(data_dicts[classify_name][:test_num])
        train_data.extend(data_dicts[classify_name][test_num:])

    print(len(train_data),len(test_data))

    return train_data,test_data 

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

    p = '/home/pre_models/electra-small/'
    config_path = p +'bert_config_tiny.json'
    checkpoint_path = p + 'electra_small'
    dict_path = p +'vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    filename = '../classify_texts/IFLYTEK/train.json'
    vec_path = 'IFLYTEK/vecs.npy'
    k_means_path = 'IFLYTEK/kMeans_data.json'

    maxlen = 512
    label_counts = 64

    datas = []
    D = fa.read_json(filename)
    for l in D:  
        text = l["sentence"]   
        label = int(l["label"])
        datas.append((text,label))

    get_vec(
        datas,         
        vec_path,
        config_path,
        checkpoint_path,
        dict_path,
        maxlen = maxlen,
        vector_name = 'cls',
        model_type = '',
        isElectra = True 
    )

    kMeans(
        vec_path,
        filename,
        label_counts,
        k_means_path
    )

    datas = fa.read(k_means_path)

    train_data,test_data  = split_train_data_by_cluster(
        datas,
        "label",
        "cluster_label",
        test_size = 0.1
    )

    fa.write_json("IFLYTEK/train_data.json",train_data)
    fa.write_json("IFLYTEK/test_data.json",test_data)

    # 根据长度 少的切掉
    