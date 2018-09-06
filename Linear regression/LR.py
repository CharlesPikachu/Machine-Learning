# 线性回归模型
# 作者: Charles
# 公众号: Charles的皮卡丘
import numpy as np


# 线性回归模型
'''
__init__:
	Input:
		None
	Output:
		None
train:
	Input:
		-x: 输入样本数据
		-y: 每个样本数据对应的目标值
	Output:
		-w: 回归系数
predict:
	Input:
		-x: 输入样本数据
		-w: 回归系数
	Ouput:
		-pred: 预测结果
'''
class LR():
	def __init__(self):
		pass
	# 基本线性回归
	def train(self, x, y):
		xMat = np.mat(x)
		yMat = np.mat(y).T
		xTx = xMat.T * xMat
		# 若xTx的行列式为0, 则xTx不可逆
		if np.linalg.det(xTx) == 0.0:
			print("[Error]: xTx matrix is singular, so fail to get w...")
			return None
		w = xTx.I * (xMat.T * yMat)
		return w
	# 预测
	def predict(self, x, w):
		pred = np.mat(x) * w
		return pred