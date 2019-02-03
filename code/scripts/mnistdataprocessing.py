import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist
import random

datasets = load_mnist('data')

train_x = datasets.train.x 
train_y = datasets.train.labels

class_wise = {}

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

best_image_label = [7,5,0,1,9,9,1,3,4,3]

# for i in range(7,8):
# 	mask = test_y == i
# 	class_wise[i] = test_x[mask]
# 	image = test_x[75]
# 	plt.imshow(image.reshape(28,-1))
# 	plt.show()

def flip(x):
	flipped = np.flip(x.reshape(28, -1),1)
	return flipped.reshape(-1)

mask = train_y == 7
class_wise_7 = train_x[mask]
class_wise_7_flipped = np.array([flip(x) for x in class_wise_7])
print ("flipped", class_wise_7_flipped.shape, class_wise_7.shape)
sym_diff_images = class_wise_7_flipped - class_wise_7
print (sym_diff_images.shape)
sym_intensities = np.array([np.mean(np.absolute(sample1)) for sample1 in sym_diff_images])
intensities = np.array([2*np.mean(sample2) for sample2 in class_wise_7])

intensity_ratios = sym_intensities/intensities

print(intensity_ratios.shape)

train_x_big_intensities = class_wise_7[intensity_ratios < 0.5]

for j in range(40):
	plt.figure(j)
	plt.imshow(train_x_big_intensities[random.randint(0,len(train_x_big_intensities)-1), :].reshape(28,-1))
	plt.show()

# print class_wise_7.shape
# intensities = np.array([np.mean(sample) for sample in class_wise_7])
# print intensities.shape
# big_intensities = intensities[intensities > 0.175]
# train_x_big_intensities = class_wise_7[intensities > 0.175]
# print big_intensities.shape


# for j in range(40):
# 	plt.figure(j)
# 	plt.imshow(train_x_big_intensities[random.randint(0,len(big_intensities)), :].reshape(28,-1))
# 	plt.show()
