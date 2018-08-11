# KNN应用实例-手写数字识别
# 作者: Charles
# 公众号: Charles的皮卡丘
import KNN
import numpy as np
from PIL import Image


def main(k=10, p=2):
	# 数据载入
	data = np.zeros((5000, 32 * 32))
	labels = np.zeros((5000, 1))
	for i in range(500):
		for j in range(10):
			labels[i*10+j, :] = j
			pic = Image.open('./dataset/%d%d.png' % (i, j))
			width = pic.size[0]
			height = pic.size[1]
			for x in range(width):
				for y in range(height):
					data[i*10+j, x*width+y] = pic.getpixel((x, y))
	data_train = data[:4500, :]
	labels_train = labels[:4500, :]
	data_test = data[4500:, :]
	labels_test = labels[4500:, :]
	# k-NN模型
	model = KNN.KNN(k, p)
	# 预测
	n_correct = 0
	n_total = 500
	for i in range(500):
		pred = model.classify(data_test[i, :], data_train, labels_train)
		print('[True Lable]: %d, [Predict Label]: %d' % (int(labels_test[i][0]), pred))
		if int(pred) == int(labels_test[i][0]):
			n_correct += 1
	acc = (n_correct / n_total) * 100
	print('[Test Accuracy]: %.2f' % acc)



if __name__ == "__main__":
	main()