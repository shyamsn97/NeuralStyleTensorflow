import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_and_scale(directory,size=128,mean=np.array([0.0, 0.0, 0.0]),
					std=np.array([1.0, 1.0, 1.0])):
    '''
        Loads, resizes, and scales image by mean and standard deviation vectors
    '''
    img = cv2.imread(directory)
    img = cv2.resize(img,(size,size))
    img = np.clip(img,0,255)
    img = img/255.0
    img = (img - mean)/std
    img = img.reshape(1,size,size,3)
    return np.clip(img,0,1)

if __name__ == "__main__":
	img = load_and_scale("images/dancing.jpg",128,np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0]))
	plt.figure()
	plt.imshow(np.squeeze(img))
	plt.show()