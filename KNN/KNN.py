# 简单的KNN模型
# 作者: Charles
# 公众号: Charles的皮卡丘
import operator
import numpy as np


# KNN模型
'''
__init__:
	Input:
		-k(int): k近邻算法中的k值
		-p(int): k近邻算法中距离度量用Lp距离，参数p决定了Lp距离公式中p的取值
	Output:
		None
classify:
	Input:
		-x(numpy.ndarray): 单个实例特征向量x
		-dataSet(numpy.ndarray): 训练集特征值矩阵
		-labels(numpy.ndarray): 训练集标签(即每个实例对应的类别)
		-is_norm(bool): 是否对数据进行归一化
	Output:
		-sortedClassCount[0][0](type(label)): 分类结果
__normalized:
	Input: 
		-data(numpy.ndarray): 未归一化的特征值矩阵
	Output: 
		-data_norm(numpy.ndarray): 归一化后的特征值矩阵
__Lp:
	Input:
		-x(numpy.ndarray): 单个实例特征向量x
		-dataSet(numpy.ndarray): 训练集特征值矩阵
		-p(int): k近邻算法中距离度量用Lp距离，参数p决定了Lp距离公式中p的取值
	Output:
		-diffMat_p_dis(numpy.ndarray): 实例特征向量x与训练集各个特征向量之间的Lp距离
'''
class KNN():
	def __init__(self, k, p):
		self.k = k
		self.p = p
	# 分类器
	def classify(self, x, dataSet, labels, is_norm=True):
		k = self.k
		p = self.p
		if is_norm:
			x = self.__normalized(x)
			dataSet = self.__normalized(dataSet)
		distances = self.__Lp(x, dataSet, p)
		# 升序排序后得到对应下标的列表
		sortedDistIdx = distances.argsort()
		classCount = {}
		for i in range(k):
			voteLabel = labels[sortedDistIdx[i]][0]
			classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
		sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
		return sortedClassCount[0][0]
	# Normalize, 归一化数字特征值到0-1范围
	def __normalized(self, data):
		data_norm = np.zeros(np.shape(data))
		if len(data.shape) == 1:
			minVal = data.min()
			maxVal = data.max()
			diff = maxVal - minVal
			data_norm = data / diff
		elif len(data.shape) == 2:
			num_data = data.shape[0]
			for i in range(num_data):
				minVal = data[i, :].min()
				maxVal = data[i, :].max()
				diff = maxVal - minVal
				data_norm[i, :] = (data[i, :] - minVal) / diff
		return data_norm
	# Lp距离计算
	def __Lp(self, x, dataSet, p):
		assert isinstance(p, int) == True
		# 数据集行数等于数据集的数据量, 即某一行数据对应训练集中某一个实例的特征向量
		diffMat = abs(np.tile(x, (dataSet.shape[0], 1)) - dataSet)
		diffMat_p = diffMat ** p
		diffMat_p_dis = diffMat_p.sum(axis=1)
		diffMat_p_dis = diffMat_p_dis ** (1 / p)
		return diffMat_p_dis