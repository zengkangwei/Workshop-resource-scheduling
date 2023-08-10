### import packages
import math
import numpy as np
import pandas as pd
import time
print('time.time()')
start = time.time()

### reading in dataset
data1 = pd.read_csv('chejian.csv')
data2 = pd.read_csv('gongzuo.csv')


# 计算资源使用率
def utl1(R, y):
    for i in range(len(y)):
        utl = 1 - (R - y) / R
    return utl


# 计算平均资源使用率
def averutl1(utl):
    averutl = np.mean(utl, axis=1)
    return averutl


def ske1(utl):
    a = len(utl)
    ske = []
    averutl = averutl1(utl)
    for i in range(a):
        utlsum = 0
        for j in range(3):
            if averutl[i, 0] != 0:
                utlsum = utlsum + (utl[i, j] / averutl[i, 0] - 1) ** 2
        if averutl[i, 0] != 0:
            ske = np.append(ske, math.sqrt(utlsum))
        else:
            ske = np.append(ske, 0)
    return ske


# 资源请求矩阵：
Req = data2[['resource1', 'resource2', 'resource3']].copy()
# 任务一的请求矩阵
Req1 = Req[0:6].values
AllReq = []
R1 = Req1.tolist()
AllReq.append(R1)
# 任务二的请求矩阵
Req2 = Req[6:10].values
R2 = Req2.tolist()
# 任务3的请求矩阵
Req3 = Req[10:14].values
R3 = Req3.tolist()
# 拼接一个请求矩阵的数组便于将任务与任务区分开
AllReq.append(R2)
AllReq.append(R3)
# 可用资源分配矩阵即策略集（三维数组）：
# 每个子任务对应一个二维数组：
# 二维数组的行代表车间分配给本任务的3种资源分别是多少；列代表不同的车间13*3*13
x = np.zeros([13, 3, 13], dtype=int, order='C')
# 权重w，和QOS参数评估值：q
w = data2[['w1', 'w2', 'w3', 'w4', 'w5']].copy().values
w1 = w[0:6]
Allw = []
w1 = w1.tolist()
Allw.append(w1)
# 任务二的w
w2 = w[6:10]
w2 = w2.tolist()
# 任务3的w
w3 = w[10:14]
w3 = w3.tolist()
Allw.append(w2)
Allw.append(w3)
# 假设q固定（这样会导致任务频繁分配在同一车间对完成分配不利
q = data1[['QOS1', 'QOS2', 'QOS3', 'QOS4', 'QOS5']].copy().values
print(q)
q1 = data1[['QOS1', 'QOS2', 'QOS3', 'QOS4', 'QOS5']].copy().values
alphard = 1.2
# 资源价格p
p = data1[['rprice1', 'rprice2', 'rprice3']].copy().values

number=0
# 分配任务
def allocation(x, AllReq, Allw):
    global number
    # 初始资源总量
    R = data1[['resource1', 'resource2', 'resource3']].copy().values

    for i in range(len(AllReq)):
        y = np.zeros([3, 13], dtype=int, order='C')
        for j in range(len(x)):
            y = np.mat(x[j]) + y
        y = y.transpose()

        #更新q1这里应该乘的都是最初的q
        for m in range(len(y)):
            q[m, 0] = (1 - ((y[m] / R[m]).sum(axis=1)) / 3) * q1[m, 0]
        # 计算平均资源分配

        utl = utl1(R, y)
        # 计算资源偏度（不均衡性）作为替换依据
        ske = ske1(utl)
        # print(ske)
        index1 = np.argsort(ske.ravel())[-1]
        averutl = averutl1(utl)
        print(averutl)
        utility = []
        # 子任务在车间的收益
        for k in range(13):
            QOS = np.dot(Allw[i], q[k])
            utility.append(QOS + (alphard - 1) * averutl[k])
        result = np.array(utility)

        # 返回数组最大值的索引
        index = np.argsort(result.ravel())[-1]
        # 选出最大值对应车间进行分配更新x
        if len(AllReq) == 6:
            n = i
        elif len(AllReq) == 4:
            n = i + 6
        else:
            n = i + 10
        a = x[n, :, index] + AllReq[i]
        b = np.array(y[index]).ravel()#这里的y是分配前的y应该再加上本次分配的
        g = 1
        # 判断资源是否用完
        for l in range(len(R[index])):
            if b[l]+a[l] < R[index][l]:
                g = 1
            else:
                g = 0
        if g == 1:
            for m in range(3):
                x[n, m, index] = a[m]
        else:
            number=number+1
            print('车间满了此子任务采用替换策略')
            print(y)
            print(ske)
    return x

#
x = allocation(x, AllReq[0], Allw[0])
x = allocation(x, AllReq[1], Allw[1])
x = allocation(x, AllReq[2], Allw[2])
x = allocation(x, AllReq[0], Allw[0])
x = allocation(x, AllReq[1], Allw[1])
x = allocation(x, AllReq[2], Allw[2])
x = allocation(x, AllReq[0], Allw[0])
x = allocation(x, AllReq[1], Allw[1])
x = allocation(x, AllReq[2], Allw[2])
x = allocation(x, AllReq[0], Allw[0])
x = allocation(x, AllReq[1], Allw[1])
x = allocation(x, AllReq[2], Allw[2])
x = allocation(x, AllReq[0], Allw[0])
x = allocation(x, AllReq[1], Allw[1])
x = allocation(x, AllReq[2], Allw[2])

# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[0])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
# x = allocation(x, AllReq[1], Allw[1])
print(number)

print(x)

end = time.time()
print(end - start, 's')