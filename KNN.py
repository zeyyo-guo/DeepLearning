import numpy as np

def createDataset():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group, labels

def classification(input, dataset, labels, k):
    datasize = dataset.shape[0]
    
    # 检查输入维度是否匹配
    if len(input) != dataset.shape[1]:
        raise ValueError("Input vector dimension does not match dataset dimension.")
    
    # 检查 k 是否为正整数且不大于数据集大小
    if k <= 0 or k > datasize:
        raise ValueError("k must be a positive integer and less than or equal to the number of samples.")
    
    # 检查 labels 长度是否与 dataset 行数一致
    if len(labels) != datasize:
        raise ValueError("Number of labels does not match the number of samples.")
    
    diff = np.tile(input, (datasize,1)) - dataset
    sqdiff = diff ** 2
    squareDist = np.sum(sqdiff, axis = 1)  
    dist = squareDist ** 0.5 
    sortdistindex = np.argsort(dist)
    classcount = {}
    for i in range(k):
        votelabel = labels[sortdistindex[i]]  
        classcount[votelabel] = classcount.get(votelabel,0) + 1  # dict.get(key, default=None)
    print(classcount)
    maxcount = 0
    for key, value in classcount.items():
        if value > maxcount:
            maxcount = value
            classes = key
    return classes


dataset, labels = createDataset()
input = [1.1, 0.3]
k = 3
output = classification(input, dataset, labels, k)
print(f'测试数据为:{input}', f'分类结果为:{output}')