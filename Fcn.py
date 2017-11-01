from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import Utils as utils
import datetime
from collections import namedtuple
import SceneParsingData as scene_parsing
import BatchDatsetReader as dataset
import time
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/camvid/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to mobile model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz'
IMAGE_NET_MEAN = [103.939, 116.779, 123.68]
IMAGE_SIZE = (320, 480)
MAX_ITERATION = int(1e5 + 1)
# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]
NUM_OF_CLASSES = 12


def mobile_net(image, final_endpoint=None, num_classes=NUM_OF_CLASSES):
    with tf.variable_scope('MobilenetV1'):
        net = image
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            for i, conv_def in enumerate(_CONV_DEFS):
                end_point_base = 'Conv2d_%d' % i
                if isinstance(conv_def, Conv):
                    net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point_base)
                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=conv_def.stride,
                                                rate=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, conv_def.depth, [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                if final_endpoint and final_endpoint == end_point_base:
                    break
    return net


def inference(image, dropout_keep_prob, num_classes=NUM_OF_CLASSES):
    print("setting up mobile initialized conv layers ...")
    mean = tf.constant(IMAGE_NET_MEAN)
    image -= mean
    net = mobile_net(image, num_classes=num_classes)

    with tf.variable_scope('inference'):
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='Conv2d_1x1')
        net = slim.convolution2d_transpose(net, num_classes, 64, 32)

        annotation_pred = tf.argmax(net, dimension=3, name="prediction")
        
    return tf.expand_dims(annotation_pred, dim=3), net


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(
        tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name="input_image")
    annotation = tf.placeholder(
        tf.int32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1], name="annotation")

    pred_annotation, logits = inference(image, dropout_keep_prob=keep_probability)
    variable_to_restore = [v for v in slim.get_variables_to_restore() if v.name.split('/')[0] == 'MobilenetV1']
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(
        annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(
        pred_annotation, tf.uint8), max_outputs=2)
    loss = utils.cal_loss(logits, annotation)
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records, test_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    train_dataset_reader = dataset.BatchDatset(
        train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(
        valid_records, image_options)
    test_dataset_reader = dataset.BatchDatset(
        test_records, image_options)

    sess = tf.Session()

    if FLAGS.mode == "train":
        print("Setting up Saver...")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(variable_to_restore)
        saver.restore(sess, 'Model_zoo/mobilenet_v1_1.0_224.ckpt')
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

        # reset saver
        saver = tf.train.Saver()
        min_valid_loss = 1e5
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(
                FLAGS.batch_size)
            feed_dict = {image: train_images,
                         annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run(
                    [loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
                

    elif FLAGS.mode == "visualize":
        saver = tf.train.Saver()
        # print_tensors_in_checkpoint_file(file_name='logs/model.ckpt-5500', tensor_name='', all_tensors=False)
        saver.restore(sess, 'logs/model.ckpt-17000')
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
            FLAGS.batch_size)
        t1 = time.time()
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        t2 = time.time()
        print(t2 - t1)
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            utils.save_image(utils.decode_segmap(valid_annotations[itr]), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            utils.save_image(utils.decode_segmap(pred[itr]), FLAGS.logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)
    elif FLAGS.mode == "validate":
        saver = tf.train.Saver()
        saver.restore(sess, 'logs/model.ckpt-99500')
        gts, preds = [], []
        for itr in xrange(116):
            test_images, test_annotations = test_dataset_reader.next_batch(
                FLAGS.batch_size
            )
            pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations,
                                                        keep_probability: 1.0})
            gt = np.squeeze(test_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt)
                preds.append(pred)
        score, class_iou = utils.scores(gts, preds, n_class=NUM_OF_CLASSES)
        for k, v in score.items():
            print('{} {}'.format(k, v))

        for i in range(NUM_OF_CLASSES):
            print('{} {}'.format(i, class_iou[i]))


if __name__ == "__main__":
    tf.app.run()
