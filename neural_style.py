import argparse 
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np

parser = argparse.ArgumentParser( #help
	formatter_class = argparse.RawTextHelpFormatter,
	description =
	'''
	Neural Style algorithm adapted from the paper "A Neural Algorithm of Artistic Style", Implemented with Tensorflow
	Usage:
		Use an input image: python neural_style.py [input image path] [content image path] [style image path] [size of images (a single number)]
		Use a white noise image: python neural_style.py "white" [content image path] [style image path] [size of images (a single number)]
	'''
)

parser.add_argument("arguments",nargs = "*")
args = parser.parse_args()

arguments = args.arguments

assert (len(arguments) == 4), "Needs 4 arguments, see -h for more info"

from src import preprocess
from src.NeuralStyle import NeuralStyle 

if arguments[0] == "white":
	input_image_path = None
else:
	input_image_path = arguments[0]

content_image_path = arguments[1]
style_image_path = arguments[2]
size = int(arguments[3])

input_img = preprocess.load_and_scale(input_image_path,size,np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0]))
content_img = preprocess.load_and_scale(input_image_path,size,np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0]))
style_img = preprocess.load_and_scale(input_image_path,size,np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0]))

ns = NeuralStyle(input_img)
mixed_image = ns.stylize(input_img,content_img,style_img)

cv2.imwrite("images/mixed_image.png",np.squeeze(mixed_image))
plt.figure()
plt.imshow(np.squeeze(mixed_image))
plt.show()

