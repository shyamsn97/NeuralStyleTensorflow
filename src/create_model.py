import numpy as np
import tensorflow as tf
import os
import h5py

def create_conv(data_dict,prev_layer,name,weights):      
    '''
        Creates a convolutional layer from the pretrained vgg network
    '''
    W = weights[0]
    b = weights[1]
    data_dict[name] = tf.nn.relu(tf.nn.conv2d(prev_layer, W, [1, 1, 1, 1], 'SAME') + b)
    return name
        
def create_avg_pool(data_dict,prev_layer,name):
    '''
        Creates an avg pool layer from the pretrained vgg network
    '''    
    data_dict[name] = tf.nn.avg_pool(prev_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return name
    
def create_vgg(include_pool,input_img):
    '''
        Creates a dictionary of tensorflow layers, copied from a pretrained VGG19 network
    '''
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    layer_names = [lay.name for lay in vgg.layers][1:]
    if os.path.isfile("vggweights.h5") == False:
        print("Creating .h5 weights file...")
        create_file(vgg)
    data_dict = {}
    data_dict["input_image"] = tf.Variable(input_img.astype("float32"),name="input_image")
    prev_name = "input_image"
    for layer_name in layer_names:
        with h5py.File('vggweights.h5','r') as hf:
            if layer_name.find("conv") != -1:
                weights = []
                weights.append(hf[layer_name + "/kernel"][:])
                weights.append(hf[layer_name + "/bias"][:])
                prev_name = create_conv(data_dict,data_dict[prev_name],layer_name,weights)
            elif np.all(layer_name.find("pool") != -1) and np.all(include_pool == True):
                prev_name = create_avg_pool(data_dict,data_dict[prev_name],layer_name)
            
    return data_dict

def create_file(vgg):
    layer_names = [lay.name for lay in vgg.layers][1:]
    with h5py.File('vggweights.h5','w') as hf:
        for layer_name in layer_names:
            if layer_name.find("conv") != -1:
                weights = vgg.get_layer(layer_name).get_weights()
                hf.create_dataset(layer_name + "/kernel",data=weights[0]) 
                hf.create_dataset(layer_name + "/bias",data=weights[1])



