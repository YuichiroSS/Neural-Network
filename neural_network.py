from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
import numpy as np
rho = 1.75

train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size = 0.20, random_state = 666)
x = train_X
y = train_y

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

w1 = np.random.randn(4, 5)
w2 = np.random.randn(5, 5)

tau = 0.01

for e in range(1000):
    loss = 0

    for p in range(len(x)):
        # xを拡張ベクトル化
        x1 = np.hstack((x[p], [1]))

        # 1層目（x1とw1を使って計算）
        g1 = sigmoid(np.dot(w1, x1))
        
        # 1層目の出力を拡張ベクトル化
        g11 = np.hstack((g1, [1]))
        
        # 2層目（g11とw2を使って計算）
        g2 = sigmoid(np.dot(w2, g11))

        # 誤差の合計を求めるため， x[p]に対する誤差を loss に加算
        loss += np.sum(1/2 * (y[p] - g2) * (y[p] - g2))
 
        # eps1を計算するため，w2の更新前に，w2を一旦保存しておく（np.copyによりデータをコピー)
        w2_old = np.copy(w2)

        # g2とy[p]を使ってeps2[0], eps2[1], eps2[2]を計算
        eps2 = (g2 - y[p])* g2 * (1 - g2)

        # w2を更新
        for k in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                w2[k, j] -= rho * eps2[k] * g11[j]
        
        # eps2 と w2_old と g1を使ってeps1[0], eps1[1], eps1[2], eps1[3]を計算
        eps1 = g1 * (1 - g1)
        for j in range(w1.shape[0]):
            tmp = 0
            for k in range(w2.shape[0]):
                # ここで，eps2とw2_oldを利用
                tmp += eps2[k] * w2_old[k][j] 
            eps1[j] *= tmp
            
        # w1を更新
        for j in range(w1.shape[0]):
            for i in range(w1.shape[1]):
                w1[j, i] -= rho * eps1[j] * x1[i]
    
    # 誤差の平均 loss/データ数 が tau 以下ならループ終了
    
    #print(loss)
    print(loss/len(x))
    if loss / len(x) < tau:
        break
        
