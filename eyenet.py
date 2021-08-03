from __future__ import print_function
import tensorflow as tf

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units
n_class = 2

# tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)

def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output
 
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output

def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)
 
def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output

def conv_net(image, gf_dim=64, reuse=False, name="conv_net"): 
    
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        c0 = relu(batch_norm(conv2d(input_ = image, output_dim = gf_dim, kernel_size = 7, stride = 1, name = 'g_e0_c'), name = 'g_e0_bn'))
        c1 = relu(batch_norm(conv2d(input_ = c0, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_e1_c'), name = 'g_e1_bn'))
        c2 = relu(batch_norm(conv2d(input_ = c1, output_dim = gf_dim * 4, kernel_size = 3, stride = 2, name = 'g_e2_c'), name = 'g_e2_bn'))
        
        r1 = residule_block_33(input_ = c2, output_dim = gf_dim*4, atrous = False, name='g_r1')
        r2 = residule_block_33(input_ = r1, output_dim = gf_dim*4, atrous = False, name='g_r2')
        r3 = residule_block_33(input_ = r2, output_dim = gf_dim*4, atrous = False, name='g_r3')
        r4 = residule_block_33(input_ = r3, output_dim = gf_dim*4, atrous = False, name='g_r4')
        r5 = residule_block_33(input_ = r4, output_dim = gf_dim*4, atrous = False, name='g_r5')
        r6 = residule_block_33(input_ = r5, output_dim = gf_dim*4, atrous = False, name='g_r6')
        r7 = residule_block_33(input_ = r6, output_dim = gf_dim*4, atrous = False, name='g_r7')
        r8 = residule_block_33(input_ = r7, output_dim = gf_dim*4, atrous = False, name='g_r8')
        r9 = residule_block_33(input_ = r8, output_dim = gf_dim*4, atrous = False, name='g_r9')
        
        return r9