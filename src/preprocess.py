import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_and_scale(directory,size=128,mean=np.array([0.0, 0.0, 0.0]),
					std=np.array([1.0, 1.0, 1.0])):
    '''
        Loads, resizes, and scales image by mean and standard deviation vectors
    '''
    if directory == None:
        img = np.random.uniform(size=(size,size,3))
    else:
        img = cv2.imread(directory)
    img = cv2.resize(img,(size,size)) 
    img = img.astype('float32')
    # Subtract the mean values
    img -= np.array([123.68, 116.779, 103.939], dtype=np.float32)
    img = img.reshape(1,size,size,3)
    return img

if __name__ == "__main__":
	img = load_and_scale("images/dancing.jpg",128,np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0]))
	plt.figure()
	plt.imshow(np.squeeze(img))
	plt.show()