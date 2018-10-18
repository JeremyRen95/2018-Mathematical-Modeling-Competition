#对每个样本特征做相关性分析
import pandas as pd
import numpy as np
import scipy.stats as stats

DeleteIndex = np.array([])

def data2mat(filename):   #将Gname恐怖组织序列标签化
    dataMat = pd.read_excel(filename)
    dataset = dataMat.drop(['gname'],axis=1).values
    gnameMat = dataMat['gname']
    gnamearray = gnameMat.values
    gnamelist = gnamearray.tolist()
    gnamesort = gnameMat.drop_duplicates().values #去重过后的标签
    gnamesort_list = gnamesort.tolist()
    gname_num = {gnamesort_list[1]:0} #初始化字典
    gnamesort = np.delete(gnamesort, 1)#所有类别列
    gnamenumMat = []
    #建立字典 组织-序号
    for i in range(gnamesort.shape[0]+1):
        if(gnamesort_list[i] == 'Unknown'):
            pass
        else:
            gname_num[gnamesort_list[i]] = i
    gname_num['Hutu extremists'] = 1
    for j in range(gnamearray.shape[0]):
        gnamenumMat.append(gname_num[gnamelist[j]])
    gnamenumMat = np.array(gnamenumMat)

    return np.c_[dataset,gnamenumMat]

def frequency(dataset,col): #dataset:array 将对应列的特征按照出现的频率标签化
    year_num = dataset.shape[0] #获取样本的个数
    year = pd.DataFrame(dataset[:,col]).drop_duplicates() #获取一共有多少年份属性
    length = len(year)
    year = year.values.reshape(1,length)[0]
    yearsum = dataset[:,col]
    year_frequency = {}
    for i in year: #求取每一年的占比
        num = len(np.nonzero(yearsum == i)[0])
        year_frequency[i] = round(num/year_num,5)
    for i in range(year_num):
        dataset[i,col] = year_frequency[dataset[i,col]]
    if (col == 27):
        zeroindex = np.nonzero(yearsum == 0.42864)[0]
        dataset[zeroindex,col] = 0
    return dataset

def test(dataset,col):    #将-9的未知量转化为其他样本对应特征的平均值
    data = dataset[:,col]
    index = np.nonzero(data != -9)[0] #获取不是-9的列索引
    index_reverse = np.nonzero(data == -9)[0] #获取不是-9的列索引
    dataverage = data[index]
    average = sum(dataverage)/len(dataverage)
    dataset[index_reverse,col] = average
    return dataset

def average(dataset,col):    #对应列特征数据归一化
    data = dataset[:, col] #获取数据集的总和
    nonanindex = np.nonzero(data == data)[0]
    datamax = max(data[nonanindex])
    datamin = min(data[nonanindex])
    rang = datamax - datamin
    dataset[:, col] = (data - datamin)/rang
    return dataset

def exchange(dataset): #替换数值函数
    victiny = dataset[:,3]
    zeroindex = np.nonzero(victiny == 0)[0]
    oneindex = np.nonzero(victiny == 1)[0]
    dataset[zeroindex,3] = 1
    dataset[oneindex,3] = 0
    propextent = dataset[:,20]
    P4 = np.nonzero(propextent == 4)[0]
    P3 = np.nonzero(propextent == 3)[0]
    P2 = np.nonzero(propextent == 2)[0]
    P1 = np.nonzero(propextent == 1)[0]
    dataset[P4, 20] = 0.25
    dataset[P3, 20] = 0.5
    dataset[P2, 20] = 0.75
    dataset[P1, 20] = 1
    return dataset

def judgement(num1,col): #num1:nkill num2:nwound  分级函数
    #{0:CeP,1:Minimal,2:Minor,3:Major,4:Catastrophic}
    level = -1
    if(col == 17):
        if(num1 == 0):
            level = 0
        elif(num1>=1 and num1<3):
            level = 1
        elif(num1>=3 and num1<10):
            level = 2
        elif (num1 >= 10 and num1 < 30):
            level = 3
        else:
            level = 4
    elif(col == 18):
        if(num1 == 0):
            level = 0
        elif(num1>=1 and num1<10):
            level = 1
        elif (num1 >= 10 and num1 < 50):
            level = 2
        elif (num1 >= 50 and num1 < 100):
            level = 3
        else:
            level = 4
    return level

def kill_wound(dataset,col):
    #首先处理死亡人数
    num = dataset.shape[0]
    data = dataset[:,col]
    level = []
    for i in range(num):
        level.append(judgement(data[i],col))
    dataset[:, col] = np.array(level)
    return dataset

#def kill_wound(dataset):  #合并伤亡人数与伤亡程度
#    data = dataset[]

def delete(dataset,DeleteIndex):   #处理经济损失特征，并将关于两个绑架特征删除
    #删除property中的-9
    data = dataset[:,19]
    todelIndex = np.nonzero(data != -9)[0]
    Index = np.nonzero(data == -9)[0]
    DeleteIndex = np.r_[DeleteIndex, Index].astype(np.int32)
    dataset = dataset[todelIndex,:]
    data = dataset[:,20]
    nonindex = np.nonzero(data != data)[0]
    dataset[nonindex,20] = 0
    dataset = np.delete(dataset,[19,21,22],axis=1)
    return dataset,DeleteIndex

def delete_nan(dataset,col,DeleteIndex): #删除对应列的空值
    data = dataset[:,col]
    nanindex = np.nonzero(data != data)[0]
    DeleteIndex = np.r_[DeleteIndex,nanindex].astype(np.int32)
    dataset = np.delete(dataset,nanindex,axis=0)
    return dataset,DeleteIndex

def delete_nine(dataset,col,DeleteIndex):
    data = dataset[:, col]
    nanindex = np.nonzero(data == -9)[0]
    DeleteIndex = np.r_[DeleteIndex, nanindex].astype(np.int32)
    dataset = np.delete(dataset, nanindex, axis=0)
    return dataset,DeleteIndex

def count(dataset): #用来获取每个特征为空值得个数
    num = dataset.shape[1] #获得特征数
    nan_count = []
    for i in range(num):
        data = dataset[:,i]
        nanindex = np.nonzero(data == -9)[0]
        count = len(nanindex) #一个特征的空值数
        nan_count.append(count)
    print(nan_count)

def pearsonr1(dataset): #判断显性相关
    r_sum = np.array([0 for i in range(dataset.shape[1])])
    p_sum = np.array([0 for i in range(dataset.shape[1])])
    num = dataset.shape[1] #得到特征的值
    for i in range(num):
        r = []
        p = []
        for j in range(num):
            r1, p1 = stats.pearsonr(dataset[:,i], dataset[:,j])
            r.append(r1)
            p.append(p1)
        r_sum = np.c_[r_sum, np.array(r)]
        p_sum = np.c_[p_sum, np.array(p)]
    return r_sum[:,1:],p_sum[:,1:]

def add_feature(dataset):
    aa = dataset[:,1]
    bb = dataset[:,7]
    cc = dataset[:, 8]
    dd = dataset[:, 10]
    ee = dataset[:, 19]
    dataset = np.delete(dataset, [1,7,8,10,19], axis=1)
    return np.c_[dataset,aa+bb+cc+dd+ee]

def add_feature2(dataset):
    aa = dataset[:, 3]
    bb = dataset[:, 4]
    cc = dataset[:, 5]
    dd = dataset[:, 2]
    ee = dataset[:, 14]
    sum = aa+bb+cc+ee+dd
    dataset[:,14] = sum
    dataset = np.delete(dataset, [2,3,4,5], axis=1)
    return dataset


aa = data2mat('附件1.xlsx') #将组织名称变为序号 并且放到最后
bb = frequency(aa,0) #年份频率
cc = frequency(frequency(bb,11),27) #组织名称频率 攻击方式频率
#dd = test(test(test(cc,26),24),23) #将INT_ANY INT_IDEO INT_LOG的-9换为平均值
#pd.DataFrame(dd).to_csv("dataset1.csv",index=False,sep=',')

ee = kill_wound(cc,18) #给死亡人数定级
ee = kill_wound(ee,17) #给受伤人数定级
ff1 = exchange(ee)
ff = exchange(ee) #给财产损失按范围定级，并将Vicinity1和0互换
ff,ii = delete(ff,DeleteIndex) #删掉property ishostkid nhostkid
ff,ii = delete_nan(ff,18,ii)
ff,ii = delete_nan(ff,17,ii)
ff,ii = delete_nan(ff,16,ii)
ff,ii = delete_nan(ff,13,ii)
ff,ii = delete_nan(ff,12,ii)
ff = np.delete(ff,14,axis=1) #删掉individual
ff = np.delete(ff,[19,20,22],axis=1) #删掉INT
ff,ii = delete_nine(ff,3,ii)
ff = add_feature(ff)
ff = np.delete(ff,9,axis=1) #删掉nality
ff = np.delete(ff,10,axis=1) #删掉nality
ff = add_feature2(ff)
rr ,pp = pearsonr1(ff)
ff = pd.DataFrame(ff)

#print(ff.corr())
ff.corr().to_csv("relative.csv",sep=',')

pd.DataFrame(rr).to_csv("rr.csv",index=False,sep=',')
pd.DataFrame(pp).to_csv("pp.csv",index=False,sep=',')
