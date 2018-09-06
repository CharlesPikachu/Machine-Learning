# 线性回归模型实例-
# 作者: Charles
# 公众号: Charles的皮卡丘
from LR import LR
import numpy as np
import matplotlib.pylab as plt


def main():
	# 随机生成数据
	x = np.random.random((50, 1))
	x.sort()
	y = np.random.random((50))
	y.sort()
	xMat = np.mat(x)
	yMat = np.mat(y)
	w = LR().train(x, y)
	xMat_copy = xMat.copy()
	y_pred = LR().predict(xMat_copy, w)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter([xMat.flatten()], [yMat.flatten()])
	ax.plot(xMat_copy, y_pred)
	plt.show()



if __name__ == '__main__':
	main()