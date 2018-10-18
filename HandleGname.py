#将预测的标签对应相应的犯罪集团
import pandas as pd
import numpy as np

def data2mat(filename):   #将Gname恐怖组织序列标签化
    dataMat = pd.read_excel(filename)
    dataset = dataMat.drop(['gname'],axis=1).values
    gnameMat = dataMat['gname']
    gnamearray = gnameMat.values
    gnamelist = gnamearray.tolist()
    gnamesort = gnameMat.drop_duplicates().values #去重过后的标签
    gnamesort_list = gnamesort.tolist()
    gname_num = {}
    gnamesort = np.delete(gnamesort, 1)#所有类别列
    gnamenumMat = []
    #建立字典 组织-序号
    for i in range(gnamesort.shape[0]+1):
        if(gnamesort_list[i] == 'Unknown'):
            pass
        else:
            gname_num[i] = gnamesort_list[i]
    print(gname_num)
    #

    return gname_num

bb = data2mat('第二问.xlsx')

data = pd.read_excel('predict.xlsx').values
num = data.shape[0]
datacol = data[:,0]
str = []
for i in range(num):
    if(datacol[i] == 0.0):
        str.append(np.nan)
    else:
        str.append(bb[int(datacol[i])])

pd.DataFrame(np.array(str)).to_csv('rubbish.csv',index=False,sep=',')


#pd.DataFrame(aa).to_csv('rubbish.csv',index=False,sep=',')
