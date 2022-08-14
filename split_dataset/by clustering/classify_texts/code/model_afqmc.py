#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

from random import random
from re import S
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

maxlen = 64
batch_size = 64

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

def load_data(fileName):
    
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = fa.read(fileName)
    # D = fa.read_json(fileName)

    
    output = []
    for l in D:

        a_text = l["sentence1"]   
        b_text = l["sentence2"]
        label = int(l["label"])
        output.append((a_text,b_text,label))

        output.append((a_text,b_text,label))
    return output

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2,label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1,text2,maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
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

final_output = Dense(2,activation='softmax')(output)
model = Model(bert.inputs, final_output)

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# datas = load_data("../AFQMC/train.json")
# import random
# random.shuffle(datas)

# train_data = [d for i, d in enumerate(datas) if i % 10 != 0]
# valid_data = [d for i, d in enumerate(datas) if i % 10 == 0]

train_data = load_data("../../split_data/AFQMC/train_data.json")
valid_data = load_data("../../split_data/AFQMC/test_data.json")

print('数据处理完成')

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

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=15,
    #     callbacks=[evaluator]
    # )

    model.load_weights('model/electra.weights')

    test_datas = load_data("../AFQMC/dev.json")
    test_generator = data_generator(test_datas, batch_size)

    score = evaluate(test_generator)
    print(score)

    # 0.706209453197405  0.7254402224281742 (32)  0.7261353104726599(64) 0.7328544949026877(16) 0.7291473586654309(8)
    # 0.7050509731232623 0.7212696941612604 0.7187210379981465s