#将恐怖组织的序号进行标签化
import pandas as pd
import numpy as np

def mklabels(filename):  #采集数据
    dataMat = pd.read_excel(filename)
    dataset = dataMat.values #滤出样本数据
    data = dataset[:,18] #选出最大的值
    gnamenum = int(max(data)+1)
    labels = np.array([0 for _ in range(gnamenum)])
    print(dataset.shape[0])
    for i in range(dataset.shape[0]):
        print(i)
        label = [0 for _ in range(gnamenum)]
        label[int(data[i])] = 1
        labels = np.c_[labels,np.array(label)]
    return labels[:,1:]



def write2csv(dataset,labels,unknowndata):
    data = pd.DataFrame(dataset)
    labels = pd.DataFrame(labels)
    unknown = pd.DataFrame(unknowndata)
    data.to_csv("dataset.csv",index=False,sep=',')
    labels.to_csv("labels.csv",index=False,sep=',')
    unknown.to_csv("unknown.csv",index=False,sep=',')

def NonanIndex(data):
    index = []
    for i in range(data.shape[0]):
        if(data[i] == 0):
            index.append(i)
    return index

def FilterData(dataset,labels,unknowndata):  #过滤掉存在nan的样本
    datasetnan = np.isnan(dataset).sum(axis=1)
    unknowndatanan = np.isnan(unknowndata).sum(axis=1)
    datasetnanindex = NonanIndex(datasetnan)
    unknowndatananindex = NonanIndex(unknowndatanan)
    return dataset[datasetnanindex,:],labels[datasetnanindex,:],unknowndata[unknowndatananindex,:]




#x_data = tf.placeholder(tf.float32,[None,29]) #特征量为29
#aa,bb,cc= data2mat('netdata.xlsx')
#dataset,labels,train = FilterData(aa,bb,cc)
#write2csv(dataset,labels,train)
#mklabels('netdata.xlsx')
pd.DataFrame(mklabels('netdata.xlsx')).to_csv('labels1.csv',index=False,sep=',')