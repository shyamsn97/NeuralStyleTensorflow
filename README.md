# NeuralStyleTensorflow

## Neural Style algorthim, as introduced in ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576), implemented in tensorflow

## How to use:
	#### Stylize an input image:
		python neural_style.py [input image path] [content image path] [style image path] [size of images] [epochs]

	#### Stylize a white noise image:
		python neural_style.py "white" [content image path] [style image path] [size of images]
