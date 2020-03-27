# coding: utf-8


from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp
from fastFM import sgd
#import matplotlib.pyplot as pltx
import time
import math
import random

ch = 0
maxy = 0
miny = 0
dt = 0
ts = 0
#输入数据与数据预处理
dataset = [[],[]]
print("Please input a name or file of a dataset:")
name = input() 
if(name == "math"):
  X = [[0.0]*3 for i in range(200)]
  dataset[1] = [0.0 for i in range(200)]
  for i in range(200):
    x = random.uniform(0,2*3.14)
    y = random.uniform(0,6.28)
    z = random.uniform(0,6.28)
    X[i][0] = x
    X[i][0] = y
    X[i][0] = z
    dataset[1][i] = (math.sin(x) + math.sin(y) + math.sin(z))*5
		#dataset[1][i] = (x + y + z)**2 
  dataset[0] = X
  fm = sgd.FMRegression(n_iter=1000, init_stdev=0.01, rank=4, l2_reg_w=0.1, l2_reg_V=0.5)
  X = sp.csc_matrix(np.array(dataset[0]), dtype=np.float128)
  Y = np.array(dataset[1], dtype=np.float64)
  fmm = fm.fit(X, Y)
	#pr_w(fmm.w_)
  X2 = sp.csc_matrix(np.array(dataset[0]), dtype=np.float128)
  y_pre = fm.predict(X2)
  s = 0.0
  for l in range(0, len(y_pre)):
    s += (y_pre[l] - dataset[1][l])**2
  print(s/len(y_pre))
elif(name == "moons"):
  dataset = datasets.make_moons(n_samples=5000)
  print(dataset[0],dataset[1])
elif(name == "circles"):
  dataset = datasets.make_circles(n_samples=5000)
#elif(name == "mushrooms"):
#  dataset = datasets.make_mushrooms(n_samples=5000)
else:
  X,y = load_svmlight_file(name)
  dataset[0] = X.toarray()
  dataset[1] = y
  d0 = dataset[1][0]
  for i in range(0, len(dataset[1])):
    if dataset[1][i] == d0:
      dataset[1][i] = 0
    else:
      dataset[1][i] = 1


		
maxl = max(dataset[1])
minl = min(dataset[1])

#对特征值进行规范化处理
minf = [0.0]*len(dataset[0][0])
maxf = [0.0]*len(dataset[0][0])
for f in range(0, len(dataset[0][0])):
    maxf[f] = dataset[0][0][f]
    minf[f] = dataset[0][0][f]
    for i in range(1, len(dataset[0])):
        maxf[f] = max(maxf[f],dataset[0][i][f])
        minf[f] = min(minf[f],dataset[0][i][f])
    if(maxf[f] > minf[f]):
        for i in range(0, len(dataset[0])):
            dataset[0][i][f] = (dataset[0][i][f] - minf[f])/(maxf[f] - minf[f])
    else:
        for i in range(0, len(dataset[0])):
            dataset[0][i][f] = 0
    maxf[f] = dataset[0][0][f]
    minf[f] = dataset[0][0][f]
    for i in range(1, len(dataset[0])):
        maxf[f] = max(maxf[f],dataset[0][i][f])
        minf[f] = min(minf[f],dataset[0][i][f])  


#对数据进行随机划分成训练数据和预测数据，分别有预测数据五组，验证(validation)数据五组
X_train = [[],[],[],[],[],[],[],[],[],[]]
y_train = [[],[],[],[],[],[],[],[],[],[]]
X_test = [[],[],[],[],[],[],[],[],[],[]]
y_test = [[],[],[],[],[],[],[],[],[],[]]
for num in range(0, 5):
    X_train[num], X_test[num], y_train[num], y_test[num] = train_test_split(dataset[0],dataset[1],test_size=0.3,random_state=num)
for num in range(5, 10):
    X_train[num], X_test[num], y_train[num], y_test[num] = train_test_split(X_train[num-5],y_train[num-5],test_size=0.3,random_state=num)
F = len(X_train[0][0])

for num in range(0,10):
    for l in range(0, len(X_train[num])):
      if y_train[num][l] == 0:
        y_train[num][l] = -1
    for l in range(0, len(X_test[num])):
      if y_test[num][l] == 0:
        y_test[num][l] = -1
#对特征进行子空间编码
def subcode(X_train, X_test, b):
        X_train_o = np.zeros((len(X_train), F*b))
        X_test_o = np.zeros((len(X_test), F*b))
        a = []
        p = []
        for f in range(0, len(X_train[0])):
            fl = []
            a.append([])
            p.append([])
            for l in range(0, len(X_train)):
                fl.append(X_train[l][f])
            for l in range(0, len(X_test)):
                fl.append(X_test[l][f])
            a[f],p[f] = np.unique(fl, return_inverse = True)
        for l in range(0, len(X_train)):
            f_en = 0
            for f in range(0, len(X_train[l])):
                    if len(a[f]) == 1:
                        f_en = f_en
                    elif len(a[f]) >= b:
                        if (X_train[l][f] - minf[f]) == (maxf[f] - minf[f]):
                            X_train_o[l][f_en + b - 1] = 1
                        else:
                            X_train_o[l][f_en + int(float(X_train[l][f] - minf[f]) / (maxf[f] - minf[f]) * b)] = 1
                        f_en += b
                    else:
                        X_train_o[l][f_en + p[f][l]] = 1
                        f_en += len(a[f])
        #if f_en < F*b:
        X_train_b = np.zeros((len(X_train), f_en ))
        for i in range(len(X_train)):
          for j in range(f_en):
            X_train_b[i][j] = X_train_o[i][j]
          #X_train_b[i][f_en] = 1
        X_train_o = X_train_b
        for l in range(0, len(X_test)):
          f_en = 0
          for f in range(0, len(X_test[l])):
            if len(a[f]) == 1:
              f_en = f_en
            elif len(a[f]) >= b:
              if (X_test[l][f] - minf[f]) == (maxf[f] - minf[f]):
                X_test_o[l][f_en + b - 1] = 1
              else:   
                X_test_o[l][f_en + int(float(X_test[l][f] - minf[f]) / (maxf[f] - minf[f]) * b)] = 1
              f_en += b
            else:
              X_test_o[l][f_en + p[f][len(X_train) + l]] = 1
              f_en += len(a[f])
        #if f_en < F*b:
        X_test_b = np.zeros((len(X_test), f_en ))
        for i in range(len(X_test)):
          for j in range(f_en):
            X_test_b[i][j] = X_test_o[i][j]
          #X_test_b[i][f_en] = 1
        X_test_o = X_test_b
	
        return X_train_o, X_test_o


		

#SEFM方法的函数
def se_fm(X_train, y_train, X_test, y_test, k, b):
    fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, rank=k, step_size = 0.01)
    X = sp.csc_matrix(np.array(X_train), dtype=np.float64)
    fmm = fm.fit(X, y_train)
	  #pr_w(fmm.w_)
    X2 = sp.csc_matrix(np.array(X_test), dtype=np.float64)
    y_pre = fm.predict(X2)
    s = 0
    for i in range(0, len(y_test)):
      if y_pre[i] * y_test[i] > 0:
        s += 1
    print("FM",'k=%d' %k, s,len(y_pre),s/len(y_pre))
    
    X_train_o, X_test_o = subcode(X_train, X_test, b)
    print(len(X_train_o[0]))
    #y_train,y_test,ts = normy(y_train, y_test, (k + 1) * len(X_train_o[0]))

    #print(ts)
    #print(y_train)
    #print(k+1,len(X_train_o[0]))
    fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, rank=k, step_size = 0.01)
    X = sp.csc_matrix(np.array(X_train_o), dtype=np.float64)
    fmm = fm.fit(X, y_train)
    print(fmm.w_)
    print(fmm.V_)
    X2 = sp.csc_matrix(np.array(X_test_o), dtype=np.float64)
    start = time.time()
    y_pre1 = fm.predict(X2)
    s = 0
    for i in range(0, len(y_test)):
      if y_pre1[i] * y_test[i] > 0:
        s += 1
    end = time.time()
    print("ODFM",'k=%d' %k,'b=%d' %b, s,len(y_pre1),s/len(y_pre1), 'time = %fs' %(end-start))
    print(fmm.w_)
    print(fmm.V_)
    
    return s/len(y_pre)


#对验证数据使用不同k值、b值、步长值的SEFM方法进行分类，找出最优的k、b、步长值（步长值设定0.01,0.05,0.1）
max_acc = 0
max_k = 0
max_b = 0
step = [0.01, 0.05, 0.1]
k = 2
for i in range(0,1):
  b = 10
  for j in range(0,1):
    pr = []
    pr_t = []
    mean_acc = 0.0
    pr.append('rank=%d b=%d'%(k,b))
    pr_t.append('rank=%d b=%d'%(k,b))
    start = time.time()
    for num in range(5,6): 
      acc = se_fm(X_train[num], y_train[num], X_test[num], y_test[num], k, b)
      mean_acc += acc
      pr.append('%.2f' %acc)
    end = time.time()
    mean_acc /= 5
    pr.append('mean=%.2f' %mean_acc)
    pr_t.append('%fs' %((end-start)/5))
	    #print(pr)
	    #print(pr_t)
    if(mean_acc > max_acc):
      max_acc = mean_acc
      max_k = k
      max_b = b
    b += 10
  k *= 2




