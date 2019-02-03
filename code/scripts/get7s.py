import numpy as np
import scripts.load_mnist
import matplotlib.pyplot as plt
from scipy import signal
import collections

data_sets = scripts.load_mnist.load_mnist('data')    

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 

train_x = data_sets.train.x 
train_y = data_sets.train.labels

nsamples = train_x.shape[0]

train_x = train_x.reshape((nsamples,input_side,input_side))

train_7s = train_x[train_y==7]

train_7s_yaxis = train_7s.sum(axis=2)

num_7s = len(train_7s)
peaks = [None]*num_7s
for i in range(num_7s):
	data = train_7s_yaxis[i]
	data = data - np.mean(data)


	window = signal.general_gaussian(5, p=1, sig=10)
	window_2 = signal.general_gaussian(2, p=1, sig=10)
	filtered = signal.fftconvolve(window, data)
	filtered = signal.fftconvolve(window_2, filtered)

	cut_off = np.max(filtered)/2.0
	filtered[filtered<cut_off]=0.0
	peakidx = signal.argrelextrema(filtered,np.greater)[0]

	# if len(peakidx)==2:
	# 	plt.clf()
	# 	plt.imshow(train_7s[i],cmap='gray')
	# 	plt.plot(data,label="data")
	# 	plt.plot(filtered,label="filtered")
	# 	plt.scatter(peakidx,filtered[peakidx])
	# 	plt.legend()
	# 	plt.savefig(str(i)+".png")
	peaks[i] = len(peakidx)

print train_7s.shape
orig_train_7 = data_sets.train.x[data_sets.train.labels == 7]
print peaks[:100]
print orig_train_7[np.array(peaks) == 2].shape

# for i in range(num_7s):
# 	if peaks[i]==3:
# 		plt.clf()
# 		plt.imshow(train_7s[i],cmap='gray')
# 		plt.plot(train_7s_yaxis[i])
# 		plt.show()
# 		break
