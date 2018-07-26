import numpy as np
import tensorflow as tf

def create_conv(data_dict,prev_layer,model_layer,model_num):      
    '''
        Creates a convolutional layer from the pretrained vgg network
    '''
    weights = model_layer.get_weights()
    data_dict["conv" + str(model_num)] = \
        tf.layers.conv2d(
            inputs=prev_layer,
            filters=model_layer.filters,
            kernel_size=model_layer.kernel_size,
            strides=model_layer.strides,
            padding=model_layer.padding,
            data_format=model_layer.data_format,
            dilation_rate=model_layer.dilation_rate,
            activation=model_layer.activation,
            use_bias=model_layer.use_bias,
            kernel_initializer= tf.constant_initializer(weights[0]),
            bias_initializer= tf.constant_initializer(weights[1]),
            kernel_regularizer=model_layer.kernel_regularizer,
            bias_regularizer=model_layer.bias_regularizer,
            activity_regularizer=model_layer.activity_regularizer,
            kernel_constraint=model_layer.kernel_constraint,
            bias_constraint=model_layer.bias_constraint,
            trainable=False,
            name="conv" + str(model_num),
            reuse=None  
        )
    return "conv" + str(model_num)
        
def create_max_pool(data_dict,prev_layer,model_layer,model_num):
    '''
        Creates a max pool layer from the pretrained vgg network
    '''    
    data_dict["pool" + str(model_num)] = \
        tf.layers.max_pooling2d(
            inputs=prev_layer,
            pool_size=model_layer.pool_size,
            strides=model_layer.strides,
            padding=model_layer.padding,
            data_format=model_layer.data_format,
            name="pool" + str(model_num)
        )
    return "pool" + str(model_num)
    
def create_vgg(include_pool,input_img):
    '''
        Creates a dictionary of tensorflow layers, copied from a pretrained VGG19 network
    '''
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    print(vgg.summary())
    data_dict = {}
    data_dict["input_image"] = tf.Variable(initial_value=input_img.astype("float32"),name="input_image")
    data_dict["input_placeholder"] = tf.placeholder(tf.float32,shape=data_dict["input_image"].shape)
    data_dict["assign_op"] = data_dict["input_image"].assign(data_dict["input_placeholder"])
    
    layers = vgg.layers
    prev_name = create_conv(data_dict,data_dict["input_image"],layers[1],1)
    layers = vgg.layers[2:]
    conv_num = 2
    max_num = 1
    
    for layer in layers:
        if layer.name.find("conv") != -1:
            prev_name = create_conv(data_dict,data_dict[prev_name],layer,conv_num)
            conv_num += 1
        elif np.all(layer.name.find("pool") != -1) and np.all(include_pool == True):
            prev_name = create_max_pool(data_dict,data_dict[prev_name],layer,max_num)
            max_num += 1
            
    return data_dict