#处理训练数据和待预测的数据 将空值转变为对应特征的平均值
import pandas as pd
import numpy as np

def perhandle(dataset): # 16,17
    data = dataset[:,16]
    zeroindex = np.nonzero(data == 0)[0]
    nonindex = np.nonzero(data == -9)[0]
    dataset[zeroindex,17] = 0
    dataset[nonindex,17] = np.nan
    dataset = np.delete(dataset,17,axis = 1)
    return dataset

def handle_nan(dataset):
    colnum = dataset.shape[1]
    for i in range(colnum-1):
        data = dataset[:,i]
        if(i == 3 or i ==4):
            notnonindex = np.nonzero(data == data)[0]
        else:
            notnonindex = np.nonzero(data >= 0)[0]
        nineindex = np.nonzero(data == -9)[0]
        nine2index = np.nonzero(data == -99)[0]
        nonindex = np.nonzero(data != data)[0]
        if(len(nonindex) == 0 and len(nineindex) == 0 and len(nine2index) == 0):
            continue
        length = len(notnonindex)
        numsum = sum(dataset[notnonindex,i])
        average = numsum/length
        dataset[nonindex, i] = average
        dataset[nineindex, i] = average
        dataset[nine2index, i] = average
    return dataset

def delete2017(dataset):
    data = dataset[:,-1]
    timeindex = np.nonzero(data > 2006)[0]
    return dataset[timeindex,:]

dataset2 = pd.read_excel('unknow.xlsx')
aa2 = handle_nan(perhandle(dataset2.values))
print(aa2.shape)
pd.DataFrame(aa2).to_csv("topredict.csv",index=False,sep=',')