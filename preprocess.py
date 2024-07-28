import numpy as np
from sklearn.model_selection import train_test_split
from numpy.fft import fft
from mne.io import concatenate_raws, read_raw_edf, read_epochs_eeglab
import os
import mne
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

way,shot,depth=5,5,64
bci_height, bci_width = 1, 2000

def get_label(epochs):
    true_label = []
    dic = {v: k for k, v in epochs.event_id.items()}
    test = epochs.events[:, -1]
    for record in epochs.events:
        if dic[record[2]] in ['139', '141', '145']:  # brake
            true_label.append(0)
        elif dic[record[2]] in ['125', '127']:  # turn
            true_label.append(1)
        elif dic[record[2]] in ['129', '131']:  # change
            true_label.append(2)
        elif dic[record[2]] in ['137', '143']:  # throttle
            true_label.append(3)
        elif dic[record[2]] in ['133']:  # stable
            true_label.append(4)
    return true_label


def find_class_index(label, c):
    y = []
    for i in range(len(label)):
        if label[i] == c:
            y.append(i)
    return y


def data_preprocess(test_size:0.2):
    DATA = np.zeros(shape=(5234, 64, 2000))
    y = []
    pointer = 0
    # '/home/Users/zwq/EEGclassify/data_arrange/'
    datapath = 'E:/markdown/code/MPDB/preprocessed data/preprocessed data/'
    # 30个样本
    for i in range(1, 31):
        EEG_temp = read_epochs_eeglab(datapath+'EEG/EEG_' + str(i) + '.set')
        EMG_temp = read_epochs_eeglab(datapath+'EMG/EMG_' + str(i) + '.set')
        GSR_temp = read_epochs_eeglab(datapath+'GSR/GSR_' + str(i) + '.set')
        label_EEG = get_label(EEG_temp)
        label_EMG = get_label(EMG_temp)
        label_GSR = get_label(GSR_temp)

        EEG = EEG_temp.get_data()
        EMG = EMG_temp.get_data()
        GSR = GSR_temp.get_data()

        EEG_index, EMG_index, GSR_index = [], [], []

        # 五种驾驶行为 XXX_index长度从1变到5
        for i in range(5):
            EEG_index.append(find_class_index(label_EEG, i))
            EMG_index.append(find_class_index(label_EMG, i))
            GSR_index.append(find_class_index(label_GSR, i))
        # 按照EEG的数量对齐构建数组
        for i in range(5):
            for j in range(min(len(EEG_index[i]), len(EMG_index[i]), len(GSR_index[i]))):
                sample = np.concatenate(
                    [EEG[EEG_index[i][j], :, :], EMG[EMG_index[i][j], :, :], GSR[GSR_index[i][j], :, :]], axis=0)
                DATA[pointer, :, :] = sample
                pointer += 1
                y.append(i)
    DATA = abs(fft(DATA))
    return train_test_split(DATA, y, test_size=.2, random_state=35)     # X_train, X_test, y_train, y_test


'''
数据读取
'''
def data_preprocess_with_all( ):
    DATA = np.zeros(shape=(5234, 64, 2000))
    y = []
    pointer = 0
    # '/home/Users/zwq/EEGclassify/data_arrange/'
    datapath = 'E:/markdown/code/MPDB/preprocessed data/preprocessed data/'
    # 30个样本
    for i in range(1, 31):
        EEG_temp = read_epochs_eeglab(datapath+'EEG/EEG_' + str(i) + '.set')
        EMG_temp = read_epochs_eeglab(datapath+'EMG/EMG_' + str(i) + '.set')
        GSR_temp = read_epochs_eeglab(datapath+'GSR/GSR_' + str(i) + '.set')
        label_EEG = get_label(EEG_temp)
        label_EMG = get_label(EMG_temp)
        label_GSR = get_label(GSR_temp)

        EEG = EEG_temp.get_data()
        EMG = EMG_temp.get_data()
        GSR = GSR_temp.get_data()

        EEG_index, EMG_index, GSR_index = [], [], []

        # 五种驾驶行为 XXX_index长度从1变到5
        for i in range(5):
            EEG_index.append(find_class_index(label_EEG, i))
            EMG_index.append(find_class_index(label_EMG, i))
            GSR_index.append(find_class_index(label_GSR, i))
        # 按照EEG的数量对齐构建数组
        for i in range(5):
            for j in range(min(len(EEG_index[i]), len(EMG_index[i]), len(GSR_index[i]))):
                sample = np.concatenate(
                    [EEG[EEG_index[i][j], :, :], EMG[EMG_index[i][j], :, :], GSR[GSR_index[i][j], :, :]], axis=0)
                DATA[pointer, :, :] = sample
                pointer += 1
                y.append(i)
    DATA = abs(fft(DATA))
    return (DATA, y)


'''
数据集划分:对way种类别分别截取数据的前shot个样本
参数    train_dataset: 对way种类别分别截取数据的前shot个样本
        test_dataset: 剩余样本
        train_y: 前shot个样本对应的标签
        test_y: 剩余样本标签
'''
def data_split(DATA, y, way:5,shot:20):
    # DATA, y = data_preprocess_with_all( )

    # 打乱数据
    shuffle = np.random.permutation(DATA.shape[0])
    DATA = DATA[shuffle]
    y = np.array(y)[shuffle]
    # 排序
    sorted_indices = np.argsort(y)
    data_sorted = DATA[sorted_indices]
    y_sorted = y[sorted_indices]
    
    train_dataset = np.zeros(shape=(1,64, 2000))
    test_dataset = np.zeros(shape=(1,64, 2000))
    train_y = np.zeros(shape=(1))
    test_y = np.zeros(shape=(1))

    for i in range(way):
        index =[]
        index = np.where(y_sorted == i)
        start = index[0][0]
        end = index[0][-1]
        train_dataset = np.concatenate( (train_dataset, np.array(data_sorted)[start:start+shot]) )
        train_y = np.concatenate( (train_y,np.array(y_sorted)[start:start+shot]) )
        test_dataset = np.concatenate( (test_dataset, np.array(data_sorted)[start+shot:end+1]) )
        test_y = np.concatenate( (test_y,np.array(y_sorted)[start+shot:end+1]) )

    train_dataset = np.delete(train_dataset, 0, axis=0)
    train_y = np.delete(train_y, 0, axis=0)[ : ,np.newaxis]
    test_dataset =  np.delete(test_dataset, 0, axis=0)
    test_y = np.delete(test_y, 0, axis=0)[ : ,np.newaxis]
    
    return train_dataset, test_dataset, train_y, test_y


DATA, y = data_preprocess_with_all( )

train_dataset, test_dataset, train_y, test_y = data_split(DATA=DATA, y=y, way=5, shot=20)
# 需要根据train阶段所需的样本个数确定slide的参数



sorted_indices = np.argsort(train_y)
train_dataset_sorted = train_dataset[sorted_indices]
train_y_sorted = train_y[sorted_indices]


for i in range(way):
    support[i] = train_dataset_sorted[train_y_sorted == i][:shot]
    query[i] = train_dataset_sorted[train_y_sorted == i][shot:]
property_pos = np.where(train_y_sorted == 0)
# support和query的划分
for i, epi_cls in enumerate(epi_classes):
    selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
    support[i] = train_dataset[epi_cls, selected[:n_shot]]
    query[i] = train_dataset[epi_cls, selected[n_shot:]]

query = x_train.shape[0]-shot
support = np.zeros([way, shot, 1, depth, bci_height, bci_width], dtype=np.float32)
query = np.zeros([way, query,  1, depth, bci_height, bci_width], dtype=np.float32)

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
y = np.array(y_train)
y = y.reshape(-1,1)
encoder.fit(y)
y_oh = encoder.transform(y).toarray() #288*1->288*4
# print(y_oh,y_oh.shape) #4187*5
one_hot_labels_tensor = torch.from_numpy(y_oh).float()
# one_hot_labels_tensor = one_hot_labels_tensor.cuda(GPU)


x_train = np.random.permutation(x_train)
# random_indices = torch.randperm(x_train.size(0))
# sort_indices = x_train[:, -1].argsort()
#     # 使用排序索引对整个矩阵进行排序
#     data_with_labels = data_with_labels_shuffle[sort_indices]



# def data_preprocess(num_s=5,way=5,shot=5):
#     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     DATA = np.zeros(shape=(5234, 64, 2000))
#     y = []
#     pointer = 0
#     # '/home/Users/zwq/EEGclassify/data_arrange/'
#     datapath = 'E:/markdown/code/MPDB/preprocessed data/preprocessed data/'
#     # 30个样本
#     for i in range(1, 31):
#         EEG_temp = read_epochs_eeglab(datapath+'EEG/EEG_' + str(i) + '.set')
#         EMG_temp = read_epochs_eeglab(datapath+'EMG/EMG_' + str(i) + '.set')
#         GSR_temp = read_epochs_eeglab(datapath+'GSR/GSR_' + str(i) + '.set')
#         label_EEG = get_label(EEG_temp)
#         label_EMG = get_label(EMG_temp)
#         label_GSR = get_label(GSR_temp)

#         EEG = EEG_temp.get_data()
#         EMG = EMG_temp.get_data()
#         GSR = GSR_temp.get_data()

#         EEG_index, EMG_index, GSR_index = [], [], []

#         # 五种驾驶行为 XXX_index长度从1变到5
#         for i in range(5):
#             EEG_index.append(find_class_index(label_EEG, i))
#             EMG_index.append(find_class_index(label_EMG, i))
#             GSR_index.append(find_class_index(label_GSR, i))

#         # 按照EEG的数量对齐构建数组
#         for i in range(5):
#             for j in range(min(len(EEG_index[i]), len(EMG_index[i]), len(GSR_index[i]))):
#                 sample = np.concatenate(
#                     [EEG[EEG_index[i][j], :, :], EMG[EMG_index[i][j], :, :], GSR[GSR_index[i][j], :, :]], axis=0)
#                 DATA[pointer, :, :] = sample
#                 pointer += 1
#                 y.append(i)
#     DATA = abs(fft(DATA))

#     # print(item_num)返回每类的个数
#     item_num = np.zeros(shape=(1,5))
#     for i in range(5):
#         item_num[0][i]=y.count(i)
#     # DATA.sort(axis=0)

#     # 确保 y 是 numpy 数组
#     y = np.array(y)
#     y = y.reshape(-1, 1)
#     DATA_reshape = DATA.reshape(5234, 64*2000)
#     data_with_labels = np.concatenate((DATA_reshape, y), axis=-1)
#     data_with_labels_shuffle = np.random.permutation(data_with_labels)

#     # 获取最后一列的排序索引
#     sort_indices = data_with_labels_shuffle[:, -1].argsort()
#     # 使用排序索引对整个矩阵进行排序
#     data_with_labels = data_with_labels_shuffle[sort_indices]

#     w_y = y.shape[-1]  # 假设你知道y的宽度
#     data_with_shuffle = data_with_labels[..., :-w_y]
#     data_with_shuffle = data_with_shuffle.reshape(5234, 64, 2000)
#     y_with_shuffle = data_with_labels[..., -w_y:]
#     y_with_shuffle = y_with_shuffle.squeeze(1)
#     y = y.tolist() 
#     return train_test_split(data_with_shuffle, y_with_shuffle, test_size=.2, random_state=35)     # X_train, X_test, y_train, y_test

