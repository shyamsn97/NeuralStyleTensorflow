import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .create_model import create_vgg

class NeuralStyle():
    
    def __init__(self,input_image):
        tf.reset_default_graph()
        self.outputs = None
        self.dict = None
        self.model_style_layers = None
        self.model_content_layers = None
        self.data_dict = create_vgg(include_pool=True,input_img=input_image)
        print(self.data_dict.keys())
    
    def content_loss(self,session,content_image,layer_list):
        
        layer_list = [self.data_dict[layer] for layer in layer_list]
        content_values = session.run(
            layer_list, feed_dict={self.data_dict["input_image"]:content_image})
        losses = []
        for gen_layer, content_val in zip(layer_list,content_values):
            losses.append(tf.reduce_mean(
                tf.square(tf.subtract(gen_layer,content_val))))
        total_loss = tf.reduce_mean(losses)
        return total_loss

    def style_loss(self,session,style_image,layer_list):
        
        layer_list = [self.data_dict[layer] for layer in layer_list]
        
        def calc_gram(feature):
            shape = feature.get_shape()
            val = tf.reshape(feature,[shape[0]*shape[1],shape[2]*shape[3]])
            gram = tf.matmul(tf.transpose(val),val)
            normalize = np.prod(shape.as_list())
            return gram/normalize
            
        gen_grams = [calc_gram(lay) for lay in layer_list]
        style_grams = session.run(
            gen_grams,feed_dict={self.data_dict["input_image"]:style_image})
        losses = []
        for gen_gram, style_gram in zip(gen_grams,style_grams):
            losses.append(tf.reduce_mean(
                tf.square(tf.subtract(gen_gram,style_gram))))
        total_loss = tf.reduce_mean(losses)
        return total_loss 
    
    def stylize(self,input_img,content_img,style_img,
                     style_layerlist = ["conv1","conv3","conv5","conv9","conv14"],
                     content_layerlist = ["conv9"],
                    style_weight = 1000.0, content_weight = 1.0,epochs=500):
        
        content_img = content_img.astype("float32")
        style_img = style_img.astype("float32")

        init = tf.global_variables_initializer()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.data_dict["assign_op"],
                     feed_dict={self.data_dict["input_placeholder"]:input_img})
            loss = tf.multiply(self.content_loss(sess,
                                     content_img,
                                     content_layerlist),content_weight) + \
                    tf.multiply(self.style_loss(sess,
                                    style_img,
                                    style_layerlist),style_weight)

            training_op = optimizer.minimize(loss,var_list=self.data_dict["input_image"])
            sess.run(tf.global_variables_initializer())
            grad = tf.gradients(loss,[self.data_dict["input_image"]])
            bar = tqdm(np.arange(epochs))
            for epoch in bar:
                bar.set_description("Total Loss: " + str(sess.run(loss)))
                sess.run(training_op)
            
            final = np.squeeze(np.clip(self.data_dict["input_image"].eval(),0,1))
            return final
