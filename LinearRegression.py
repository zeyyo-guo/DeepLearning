import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

xi = np.array([162,165,159,173,157,175,161,164,172,158])
yi = np.array([48,64,53,66,52,68,50,52,64,49])

# 定义线性函数
def func(p,x):
    a,b = p
    return a*x + b

# 计算残差
def residuals(p,x,y):
    return func(p,x)-y

p0 = [1,20]
# 使用最小二乘法拟合数据
plsq = least_squares(residuals, p0, args=(xi,yi))
a,b = plsq.x
print("a =",a,"b =",b)

# 计算拟合值与实际值的残差
xy_res = func(plsq.x,xi) - yi
print('residuals:',xy_res)

# 计算残差的平方和
xy_ressum = np.dot(xy_res,xy_res)
print('sum of residuals:',xy_ressum)

# 指定图像比例为8:6
plt.figure(figsize = (8,6))

# 绘制散点图和拟合直线
# plt.scatter(xi, yi, color='r',marker='o',s=100)
# plt.xlabel('Height:cm')
# plt.ylabel('Weight:kg')
# x = np.linspace(150,180,100)
# y = a*x+b
# plt.plot(x,y,color='b')

# 绘制残差的QQ图
ax = plt.subplot(1,1,1)
fig = qqplot(np.array(xy_res),line='q',ax=ax)
plt.show()