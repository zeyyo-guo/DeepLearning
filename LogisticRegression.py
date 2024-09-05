# 逻辑回归的步骤：发现逻辑函数： sigmoid函数；构造损失函数：交叉熵；求解成本函数：梯度下降法；评价：AUC、ROC曲线

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
data = iris.data
target = iris.target
# print(iris)
x = data[0:100,[0,2]]
y = target[0:100]

label = np.array(y)
index_0 = np.where(label == 0)
plt.scatter(x[index_0,0], x[index_0,1],marker='x',color = 'b',label = '0', s = 15)
index_1 = np.where(label == 1)
plt.scatter(x[index_1,0],x[index_1,1],marker='o',color = 'r', label = '1',s = 15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('scatter of X1,X2')
plt.legend()
plt.show()

class logistic(object):
    def __init__(self):
        self.W = None
    def train(self,x,y,learn_rate = 0.01,num_iters = 5000):
        num_train, num_features = x.shape
        self.W = 0.0001 * np.random.randn(num_features,1).reshape((-1,1))
        loss = []
        for i in range(num_iters):
            error, dW = self.compute(x,y)
            self.W += -learn_rate * dW
            loss.append(error)
            if i % 200 == 0:
                print("i = %d, error = %f" % (i,error))
        return loss
    def compute(self,x,y):
        num_train = x.shape[0]
        h = self.output(x)
        loss = -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        loss = loss / num_train
        dW = x.T.dot(h-y) / num_train
        return loss, dW
    def output(self,x):
        g = np.dot(x, self.W)
        return self.sigmoid(g)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def pred(self,x_test):
        h = self.output(x_test)
        y_pred = np.where(h>=0.5,1,0)
        return y_pred
    
y = y.reshape((-1,1))
one = np.ones((x.shape[0],1))
x_train = np.hstack((one,x))

classify = logistic()
loss = classify.train(x_train,y)
print(classify.W)
plt.plot(loss)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


plt.scatter(x[index_0,0], x[index_0,1],marker='x',color = 'b',label = '0', s = 15)
plt.scatter(x[index_1,0],x[index_1,1],marker='o',color = 'r', label = '1',s = 15)
x1 = np.arange(4.0,7.5,0.5)
x2 = (-classify.W[0]-classify.W[1] * x1) / classify.W[2]
plt.plot(x1,x2,color = 'r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')
plt.show()




