#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

from random import random
from unicodedata import category
import numpy as np
from keras.layers import *
from keras.models import *
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import fairies as fa
from tqdm import tqdm 
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_gelu('tanh')  # 切换gelu版本

maxlen = 48
batch_size = 96

p = '/home/pre_models/chinese-roberta-wwm-ext-tf/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# p = '/home/pre_models/electra-small/'
# config_path = p +'bert_config_tiny.json'
# checkpoint_path = p + 'electra_small'
# dict_path = p +'vocab.txt'
# tokenizer = Tokenizer(dict_path, do_lower_case=True)

category = set()

def load_data(fileName):
    
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = fa.read(fileName)

    output = []

    for l in D:  
        text = l["sentence"]   
        label = l["label"]
        category.add(label)
        output.append((text,label))

    return output

import random
random.seed(0)

# datas = load_data("../TNEWS/train.json")
# random.shuffle(datas)

# train_data = [d for i, d in enumerate(datas) if i % 10 != 0]
# valid_data = [d for i, d in enumerate(datas) if i % 10 == 0]

train_data = load_data("../../split_data/TNEWS/train_data.json")
valid_data = load_data("../../split_data/TNEWS/test_data.json")
random.shuffle(train_data)
random.shuffle(valid_data)

print('数据处理完成')

category = list(category)
category.sort()

id2label,label2id = fa.label2id(category)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text,label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text,maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label2id[label]])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 加载预训练模型
bert = build_transformer_model(
    config_path,
    checkpoint_path
    # model='electra',
)

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.output)

final_output = Dense(len(category),activation='softmax')(output)
model = Model(bert.inputs, final_output)

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(9e-6),  # 用足够小的学习率
    metrics=['accuracy'],
)

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('model/electra.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, 0)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs = 8,
        callbacks=[evaluator]
    )

    model.load_weights('model/electra.weights')

    category = set()
    test_datas = load_data("../TNEWS/dev.json")
    test_generator = data_generator(test_datas, batch_size)

    score = evaluate(test_generator)
    print(score)

    # maxlen 48
    # 8 k-means 
    # 16 k-means 
    # 32 k-means 
    # 64 k-means 0.5733
    # 128 k-means 0.5812
    # 256 k-means 0.5755
    # 512 k-means 
    
    # 0.5734 0.5744