import tensorflow as tf
from tensorflow.python.framework import ops
from glob import glob
import random
import scipy.misc



def get_data():
    image_list = glob("data/cat/*.jpg")
    image_list = image_list + glob("data/dog/*.jpg")

    random.shuffle(image_list)

    labels = []
    for image_name in image_list:
        if "cat" in image_name:
            labels.append([1,0])
        else:
            labels.append([0,1])

    tf_image_list = ops.convert_to_tensor(image_list, dtype=tf.string, name="tf_image_list")
    tf_labels = ops.convert_to_tensor(labels, dtype=tf.int32, name="tf_labels")
    return tf_image_list, tf_labels


def preprocess_image_tensor(image_tf, image_length, image_width):
    image = tf.image.convert_image_dtype(image_tf, dtype=tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, image_length, image_width)
    image = tf.image.per_image_standardization(image)
    return image


def distort_image(image_buffer):
    image = tf.image.random_brightness(image_buffer, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, 0.1, 0.9, seed=None)
    tf.image.random_brightness(image, 0.2, seed=None)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.transpose_image(image)

    return image


def get_batch(batch_size, image_length, image_width):

    print "loading image file names and labels"
    image_paths_tf, labels_tf = get_data()

    print "getting tensor slice"
    image_path_tf, label_tf = tf.train.slice_input_producer([image_paths_tf, labels_tf], shuffle=False,
                                                            name="image filename slice input producer")

    print "creating image buffer"
    image_buffer_tf = tf.read_file(image_path_tf, name="image_buffer")
    image_tf = tf.image.decode_jpeg(image_buffer_tf, channels=3, name="image")
    image_tf = preprocess_image_tensor(image_tf, image_length, image_width)

    print "creating distorted image"
    distorted_image_tf = distort_image(image_tf)

    print "making batch"
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    images_batch_tf, labels_batch_tf = tf.train.shuffle_batch_join([[image_tf, label_tf],
                                                                    [distorted_image_tf, label_tf]],
                                                                   batch_size=batch_size, capacity=capacity,
                                                                   min_after_dequeue=min_after_dequeue,
                                                                   name="image_batch")

    print "completed batch"
    return images_batch_tf, labels_batch_tf

def test_batch():
    tf.global_variables_initializer()
    with tf.Session() as sess:

        image_batch_tf, labels_batch_tf = get_batch(5)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_batch, labels_batch = sess.run([image_batch_tf, labels_batch_tf])

        for i in range(5):
            scipy.misc.imsave("test_output/out/image_" + str(i) + ".jpg", image_batch[i,:])
        print labels_batch

        coord.request_stop()
        coord.join(threads)