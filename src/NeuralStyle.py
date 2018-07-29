import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .create_model import create_vgg

class NeuralStyle():
    '''
        Neural Style algorithm adapted from the paper "A Neural Algorithm of Artistic Style", Implemented with Tensorflow
        Neural Style object
        Parameters:
            inputs: takes in an input image
            data_dict: dictionary of layers and weights from vgg19        
    '''
    def __init__(self,input_image):
        print("Loading in VGG19 Network")
        self.data_dict = create_vgg(include_pool=True,input_img=input_image)
        print("Done")
    
    def content_loss(self,session,content_image,layer_list):
        # Content loss : mean squared error between feature maps of content image and generated image
        session.run(tf.global_variables_initializer())
        layer_list_vals = [self.data_dict[layer] for layer in layer_list]
        session.run(self.data_dict["input_image"].assign(content_image))
        content_values = []
        for layer_name in layer_list:
            content_values.append(session.run(self.data_dict[layer_name]))

        losses = []
        for gen_layer, content_val in zip(layer_list_vals,content_values):
            losses.append(tf.reduce_mean(
                tf.square(tf.subtract(gen_layer,content_val))))
        total_loss = tf.reduce_mean(losses)
        return total_loss

    def style_loss(self,session,style_image,layer_list):
        # Style loss : sum of error between gram matrices of style image and generated image
        session.run(tf.global_variables_initializer())
        def gram_matrix(x):
            # gram matrix
            x = tf.reshape(x, (-1, x.shape[3]))
            return tf.matmul(tf.transpose(x), x) 

        gen_grams =  [gram_matrix(self.data_dict[layer_name]) for layer_name in layer_list]
        session.run(self.data_dict["input_image"].assign(style_image))
        total_loss = 0
        for i in range(len(layer_list)):
            style_feature = tf.convert_to_tensor(session.run(self.data_dict[layer_list[i]]))
            stlye_gram = gram_matrix(style_feature)
            total_loss += (1. / (4 * (
                (style_feature.shape[1].value * style_feature.shape[2].value)**2) * \
                style_feature.shape[3].value**2) * \
                tf.reduce_sum(tf.square(tf.subtract(gen_grams[i],stlye_gram))))

        return total_loss
    
    def stylize(self,input_img,content_img,style_img,
                     style_layerlist = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"],
                     content_layerlist = ["block4_conv1"],
                     style_weight = 1e5, content_weight = 1e0,tv_weight=1e-4,epochs=500):
        """
            Main style function
                Parameters:
                    input_img: image to optimize over
                    content_img: content image
                    style_img: style image
                    style_layerlist = list of style layers
                    content_layerlist = list of content layers
                    style_weight, content_weight, tv_weight = weight for style loss, 
                                                content loss, and total variation loss
                    epochs = number of epochs to train
        """
        content_img = tf.constant(content_img, dtype=tf.float32, name='content_img')
        style_img = tf.constant(style_img, dtype=tf.float32, name='content_img')

        optimizer = tf.train.AdamOptimizer(learning_rate=1)
        with tf.Session() as sess:
            loss = self.content_loss(sess,
                                     content_img,
                                     content_layerlist)*content_weight + \
                    self.style_loss(sess,
                                    style_img,
                                    style_layerlist)*style_weight + \
                    tv_weight * tf.image.total_variation(self.data_dict['input_image'])

            training_op = optimizer.minimize(loss,var_list=self.data_dict["input_image"])
            sess.run(tf.global_variables_initializer())
            sess.run(self.data_dict["input_image"].assign(input_img))

            orig = input_img
            print("Optimizing...")
            bar = tqdm(np.arange(epochs))
            for epoch in bar:
                bar.set_description("Total Loss: " + str(sess.run(loss)))
                sess.run(training_op)
            
            final = np.squeeze(sess.run(self.data_dict["input_image"]))
            print("L2 distance between original and generated image: " + str(np.linalg.norm(final - orig)))
            return final



