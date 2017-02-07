import tensorflow as tf

image_width, image_height = 227, 227

# placeholder for feeding batches of 3-channel images
x = tf.placeholder(tf.float32, [None, image_height, image_width, 3])

"""
    Convolutional NN modeled after CIFAR-10 but with output of 1x1x2 (for cat vs dog)
        http://cs231n.github.io/convolutional-networks/

    Structure:
        input -> conv -> relu -> pool -> fc

    Parameters:
        images of 32x32x3
        12 filters
        pool down to [16, 16, 12]


"""
def build_layers(x):

    # stride of 4

    num_kernels = [96, 256, 384, 384, 256]

    # 96 filters of 11 x 11 x 3
    w1 = tf.Variable(tf.truncated_normal([image_width, image_height, 3, num_kernels[0]], stddev=0.1))
    b1 = tf.Variable(tf.ones([num_kernels[0]])/2)

    # 256 filters of 5x5x48
    w2 = tf.Variable(tf.truncated_normal([image_width, image_height, num_kernels[0], num_kernels[1]], stddev=0.1))
    b2 = tf.Variable(tf.ones([num_kernels[1]]) / 2)

    # 384 filters of 3 x 3 x 256
    w3 = tf.Variable(tf.truncated_normal([image_width, image_height, num_kernels[1], num_kernels[2]], stddev=0.1))
    b3 = tf.Variable(tf.ones([num_kernels[2]]) / 2)

    # 384 filters of 3 x 3 x 192
    w4 = tf.Variable(tf.truncated_normal([image_width, image_height, num_kernels[2], num_kernels[3]], stddev=0.1))
    b4 = tf.Variable(tf.ones([num_kernels[3]]) / 2)

    # 256 filters of 3 x 3 x 192
    w5 = tf.Variable(tf.truncated_normal([image_width, image_height, num_kernels[3], num_kernels[4]], stddev=0.1))
    b5 = tf.Variable(tf.ones([num_kernels[4]]) / 2)





    return