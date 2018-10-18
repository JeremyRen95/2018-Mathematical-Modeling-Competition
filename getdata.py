#计算每个样本权重 然后将超过11万的数据进行分级
import pandas as pd
import numpy as np
import scipy.stats as stats

weaptype = {1.0:8,2.0:8,3.0:9,4.0:10,5.0:5,6.0:7,7.0:3,8.0:4,9.0:3,10.0:3,11.0:3,12.0:np.nan,13.0:np.nan} #7,13
attacktype = {1.0:8,2.0:8,3.0:10,4.0:7,5.0:4,6.0:5,7.0:7,8.0:2,9.0:np.nan} #9
targetype = {1.0:6,2.0:9,3.0:8,4.0:10,5.0:4,6.0:7,7.0:8,8.0:7,9.0:6,10.0:5,11.0:7,12.0:5,13.0:3,14.0:2,15.0:8,16.0:4,17.0:5,18.0:3,19.0:4,20.0:np.nan,21.0:6,22.0:6,23:np.nan} #22
DeleteIndex = np.array([])

def data2mat(filename):   #将Gname恐怖组织序列标签化
    dataMat = pd.read_excel(filename)
    dataset = dataMat.drop(['gname'],axis=1).values
    gnameMat = dataMat['gname']
    gnamearray = gnameMat.values
    gnamelist = gnamearray.tolist()
    gnamesort = gnameMat.drop_duplicates().values #去重过后的标签
    gnamesort_list = gnamesort.tolist()
    gname_num = {gnamesort_list[1]:0}
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
    if(col == 14):
        if(num1 == 0 or num1 != num1):
            level = 0
        elif(num1>=1 and num1<3):
            level = 1
        elif(num1>=3 and num1<10):
            level = 2
        elif (num1 >= 10 and num1 < 30):
            level = 3
        else:
            level = 4
    elif(col == 15):
        if(num1 == 0 or num1 != num1):
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

def delete_nan(dataset,col): #删除对应列的空值
    data = dataset[:,col]
    nanindex = np.nonzero(data != data)[0]
    #DeleteIndex = np.r_[DeleteIndex,nanindex].astype(np.int32)
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

#################################
def property_extend(dataset):
    data_property = dataset[:,16] #取出property列
    nineIndex = np.nonzero(data_property == -9)[0] #取出-9的索引
    nonineIndex = np.nonzero(data_property == 0)[0] #取出为零的索引
    dataset[nineIndex, 17] = np.nan #若为-9 则在损失金额列置空
    dataset[nonineIndex, 17] = 0#若为0 则在损失金额列置0
    return dataset

def add(dataset):
    aa = dataset[:, 1]
    bb = dataset[:, 3]
    cc = dataset[:, 4]
    dd = dataset[:, 5]
    ee = dataset[:, 6]
    ff = dataset[:, 7]
    hh = dataset[:, 8]
    ii = dataset[:, 10]
    jj = dataset[:, 18]
    sum = aa + bb + cc + dd + ee + ff + hh + ii + jj
    index_minus = np.nonzero(sum < 0)[0]
    sum[index_minus] = np.nan
    return np.c_[dataset, sum]

def link(dataset):
    attack = dataset[:,11]
    target = dataset[:,12]
    weap = dataset[:,13]
    for i in range(len(attack)):
        attack[i] = attacktype[attack[i]]
    dataset[:, 11] = attack
    for i in range(len(target)):
        target[i] = targetype[target[i]]
    dataset[:, 12] = target
    for i in range(len(weap)):
        weap[i] = weaptype[weap[i]]
    dataset[:, 13] = weap
    return dataset

def reverse(dataset): #将vicnity取反
    data = dataset[:,3]
    zeroindex = np.nonzero(data == 0)[0]
    oneindex = np.nonzero(data == 1)[0]
    dataset[zeroindex,3] = 1
    dataset[oneindex, 3] = 0
    return dataset

def evalution(dataset):
    datanum = dataset.shape[0] #获取行数
    Mi = np.array([0,0,0,0,0,0])
    Pi = np.array([0,0,0,0,0,0])
    fre_max = max(dataset[:,9])
    for i in range(datanum):
        data = dataset[i,:] #拿到一个样本的数据
        if(data[4] == data[4]):
            m1 = max(data[6],data[7])*data[4]/40
            p1 = 0
        else:
            m1 = 0
            p1 = 0.346
        if(data[8] == data[8]):
            m2 = data[8]/4
            p2 = 0
        else:
            m2 = 0
            p2 = 0.263 #待定
        m3 = data[0]
        p3 = 0
        m4 = data[1]
        p4 = 0
        if(data[3] == data[3] and data[5] == data[5]):
            m5 = ((data[2]+1)*(data[3]+data[5]))/40
            p5 = 0
        else:
            m5 = 0
            p5 = 0.106
        if(data[10] == data[10]):
            m6 = (data[10] * data[9])/(9*fre_max)
            p6 = 0
        else:
            m6 = 0
            p6 = 0.213 #待定
        M = np.array([m1,m2,m3,m4,m5,m6])
        P = np.array([p1,p2,p3,p4,p5,p6])
        Mi = np.c_[Mi,M]
        Pi = np.c_[Pi,P]
    W = np.array([0.346,0.263,0.035,0.035,0.106,0.213])
    Mi = Mi[:,1:]
    Pi = Pi[:, 1:].transpose()
    result1 = np.matmul(W,Mi)
    result2 = 1/(1-np.sum(Pi,axis=1))
    result = result1*result2
    pd.DataFrame(Mi).to_csv("M.csv", index=False, sep=',')
    pd.DataFrame(result1).to_csv("mul.csv", index=False, sep=',')
    pd.DataFrame(result2).to_csv("sum.csv", index=False, sep=',')
    pd.DataFrame(result).to_csv("result.csv", index=False, sep=',')
    return result

def level(data):
    num = data.shape[0]
    level = []
    for i in range(num):
        lev = data[i]
        if(lev>=0 and lev<0.1756):
            level.append(5)
        elif(lev>=0.1756 and lev<0.3512):
            level.append(4)
        elif (lev >= 0.3512 and lev < 0.5267):
            level.append(3)
        elif (lev >= 0.5267 and lev < 0.7023):
            level.append(2)
        elif (lev >= 0.7023):
            level.append(1)
    level_array = np.array(level)
    pd.DataFrame(level_array).to_csv("level.csv", index=False, sep=',')

def delete_nan1(dataset):
    featherNum = dataset.shape[1]
    for i in range(featherNum):
        data = dataset[:,i]
        nanindex = np.nonzero(data != data)[0]
        dataset = np.delete(dataset, nanindex, axis=0)
    return dataset

#将年份 地区 犯罪集团名称频率化
aa = data2mat('附件1b.xlsx') #将组织名称变为序号 并且放到最后
bb = frequency(aa,0) #年份频率
bb = frequency(bb,2)
bb = frequency(bb,19) #组织名称频率 攻击方式频率
cc = kill_wound(bb,14) #nkill和wound分级
cc = kill_wound(cc,15)
dd = property_extend(cc) #处理金额
dd = reverse(dd)
#ff = np.delete(dd,16,axis=1) #删除相应列
#ff = delete_nan1(ff)
#rr ,pp = pearsonr1(ff)
#pd.DataFrame(ff).to_csv("19维.csv",index=False,sep=',')
#pd.DataFrame(ff).corr().to_csv("relative.csv",index=False,sep=',')
#pd.DataFrame(rr).to_csv("rr.csv",index=False,sep=',')
#pd.DataFrame(pp).to_csv("pp.csv",index=False,sep=',')

ee = add(dd) #将相关布尔值相加
ff = link(ee) #根据表分级
ff = np.delete(ff,[1,3,4,5,6,7,8,10,16,18],axis=1) #删除相应列
res = evalution(ff)
level(res)
pd.DataFrame(ff).to_csv("ff.csv",index=False,sep=',')



