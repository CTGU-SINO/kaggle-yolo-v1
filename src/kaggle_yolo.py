import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import re
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 19
random.seed = seed
np.random.seed = seed

TRAIN_PATH = '../input/stage1_train'
TEST_PATH = '../input/stage1_test'

IMG_WIDTH = 112
IMG_HEIGHT = 112
IMG_CHANNELS = 3

KERNEL_SIZE = 3
CON2D_LAYER1 = 16
CON2D_LAYER2 = 32
CON2D_LAYER3 = 64
CON2D_LAYER4 = 128

INITIALIZER_CON2D = tf.contrib.layers.xavier_initializer_conv2d()
INITIALIZER_FULLY = tf.contrib.layers.xavier_initializer()

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

train_images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
test_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

GRID_DIM = 7
GRID_PIX = IMG_WIDTH // GRID_DIM
BATCH_SIZE = 14
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 1000000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./yolo_model/"
CHECK_POINT = './yolo'
MODEL_NAME = "yolo_model"
TOWER_NAME = 'tower'


def re_build_size():
    print('resize train images... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + "/" + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        train_images[n] = img

    # Get and resize test images
    sizes_test = []
    print('resize test images ... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + "/" + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        test_images[n] = img

    print('Done!')


def store_bounding_boxes(img, train_id, mask_id, rotby_90):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours = cv2.findContours(thresh.astype(np.uint8), 1, 2)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    x = x * (IMG_WIDTH / img.shape[1])
    w = w * (IMG_WIDTH / img.shape[1])
    y = y * (IMG_WIDTH / img.shape[0])
    h = h * (IMG_WIDTH / img.shape[0])

    if (x > IMG_WIDTH - 1):
        x = IMG_WIDTH - 1
    if (y > IMG_HEIGHT - 1):
        y = IMG_HEIGHT - 1
    if (x + w > IMG_WIDTH - 1):
        w = IMG_WIDTH - 1 - x
    if (y + h > IMG_HEIGHT - 1):
        h = IMG_HEIGHT - 1 - y

    bbdict = {"train_id": train_id, "mask_id": mask_id, "rotby_90": rotby_90, "x": x, "y": y, "w": w, "h": h}
    return bbdict


path_bboxes_csv = "../input/bboxes.csv"
if not os.path.isfile(path_bboxes_csv):
    bboxes = pd.DataFrame(columns=["train_id", "mask_id", "rotby_90", "x", "y", "w", "h"])
    row_count = 1
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + "/" + id_
        for mask_id, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
            mask_ = imread(path + '/masks/' + mask_file)
            for r in range(4):
                bboxes.loc[row_count] = store_bounding_boxes(np.rot90(mask_, r), id_, mask_id, r)
                row_count += 1
    bboxes.to_csv(path_bboxes_csv, index=False)
else:
    bboxes = pd.read_csv(path_bboxes_csv)

train_ids_df = pd.DataFrame(columns=["idx", "id_"])
cnt = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    train_ids_df.loc[cnt] = {"idx": n, "id_": id_}
    cnt += 1

train_ids_df = train_ids_df.set_index(['idx'])

bboxes['grid_row'] = bboxes['y'] // GRID_PIX
bboxes['grid_column'] = bboxes['x'] // GRID_PIX

bboxes['grid_center_x'] = bboxes['grid_column'] * GRID_PIX + GRID_PIX / 2
bboxes['grid_center_y'] = bboxes['grid_row'] * GRID_PIX + GRID_PIX / 2

bboxes['box_center_x'] = bboxes.x + bboxes['w'] / 2
bboxes['box_center_y'] = bboxes.y + bboxes['h'] / 2

bboxes['new_x'] = (bboxes.box_center_x - bboxes.grid_center_x) / (IMG_WIDTH)
bboxes['new_y'] = (bboxes.box_center_y - bboxes.grid_center_y) / (IMG_HEIGHT)

bboxes['new_w'] = np.sqrt(bboxes.w / (IMG_WIDTH))
bboxes['new_h'] = np.sqrt(bboxes.h / (IMG_WIDTH))

bboxes['confidence'] = 1

bboxes['box_area'] = bboxes.new_w * bboxes.new_h

mask_count = 1
# Set maximum bounding boxes allowed per grid cell
MAX_BB_CNT = 2


def get_grid_info(tr_id, rotby_90):
    df = bboxes.loc[(bboxes.train_id == tr_id) & (bboxes.rotby_90 == rotby_90), 'grid_row':'box_area']
    df.drop(['grid_center_x', 'grid_center_y', 'box_center_x', 'box_center_y', ], axis=1, inplace=True)
    df = df.sort_values(['grid_column', 'grid_row', 'box_area'], ascending=False)
    # print(len(df))
    global mask_count
    mask_count += len(df)
    label_info = np.zeros(shape=(GRID_DIM, GRID_DIM, MAX_BB_CNT, 5), dtype=np.float32) + 0.000001

    for ind, row in df.iterrows():
        i = int(row[0])
        j = int(row[1])
        for b in range(MAX_BB_CNT):
            if (label_info[i, j, b][4] != 1.0):
                label_info[i, j, b] = np.array(row[2:7])
                break
    return label_info


def get_labels(counts, rotations):
    grid_info = np.zeros(shape=(BATCH_SIZE, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5), dtype=np.float32)
    for i, c in enumerate(counts):
        tr_id = train_ids_df.loc[c, 'id_']
        grid_info[i] = get_grid_info(tr_id, rotations[i])
    grid_info = np.reshape(grid_info, newshape=[BATCH_SIZE, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])
    return grid_info


def get_images(counts, rotations):
    images = np.zeros(shape=(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
    for i, c in enumerate(counts):
        images[i] = np.rot90(train_images[c], rotations[i])
    return images


def next_batch():
    rotations = []
    rand_counts = []
    for i in range(BATCH_SIZE):
        rotations.append(random.randint(0, 3))
        rand_counts.append(random.randint(0, 669))
    return get_images(rand_counts, rotations), get_labels(rand_counts, rotations)


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def process_logits(logits, name=None):
    net = tf.reshape(logits, [-1, GRID_DIM * 1, GRID_DIM * 1, MAX_BB_CNT * 5 * 16, 1])
    net = tf.layers.average_pooling3d(net, [1, 1, 16], [1, 1, 16], padding="valid")

    net = tf.reshape(net, [-1, GRID_DIM * GRID_DIM * MAX_BB_CNT, 5])  # GRID_DIM = 12
    net = tf.transpose(net, [1, 2, 0])

    logits_tensor = tf.map_fn(lambda x:
                              tf.stack([
                                  tf.tanh(x[0]),
                                  tf.tanh(x[1]),
                                  tf.sqrt(tf.sigmoid(x[2])),
                                  tf.sqrt(tf.sigmoid(x[3])),
                                  tf.sigmoid(x[4])
                              ]), net)

    logits_tensor = tf.transpose(logits_tensor, [2, 0, 1])
    logits_tensor = tf.reshape(logits_tensor, [-1, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])

    return logits_tensor


def normalize_yolo_loss(labels, processed_logits, lambda_coords, lambda_noobj):
    yolo_loss = tf.reduce_sum(tf.squared_difference(labels, processed_logits), axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)
    yolo_loss = tf.reduce_sum(yolo_loss, axis=0)

    yolo_loss = tf.stack([tf.multiply(lambda_coords, yolo_loss[0]),
                          tf.multiply(lambda_coords, yolo_loss[1]),
                          yolo_loss[2],
                          yolo_loss[3],
                          tf.multiply(lambda_noobj, yolo_loss[4])])
    yolo_loss = tf.reduce_sum(yolo_loss)

    return yolo_loss


def conv_op(x, name, n_out, training, useBN, kh=5, kw=5, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=INITIALIZER_CON2D)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        con2d = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(con2d, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)

        _activation_summary(z)
    return z


def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=INITIALIZER_FULLY)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        _activation_summary(fc)

    return fc


def net(x):
    con2d1_1 = conv_op(x, "Con2d_layer1_1", CON2D_LAYER1, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    con2d1_2 = conv_op(con2d1_1, "Con2d_layer1_2", CON2D_LAYER1, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    max_pool1 = max_pool_op(con2d1_2, 'max_pooling1')  # 56*56*16
    con2d2_1 = conv_op(max_pool1, "Con2d_layer2_1", CON2D_LAYER2, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    con2d2_2 = conv_op(con2d2_1, "Con2d_layer2_2", CON2D_LAYER2, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)  # 56*56*32
    max_pool2 = max_pool_op(con2d2_2, 'max_pooling2')  # 28*28*32
    con2d3_1 = conv_op(max_pool2, "Con2d_layer3_1", CON2D_LAYER3, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    con2d3_2 = conv_op(con2d3_1, "Con2d_layer3_2", CON2D_LAYER3, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)  # 28*28*64
    max_pool3 = max_pool_op(con2d3_2, 'max_pooling3')  # 14*14*64
    con2d4_1 = conv_op(max_pool3, "Con2d_layer4_1", CON2D_LAYER4, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    con2d4_2 = conv_op(con2d4_1, "Con2d_layer4_2", CON2D_LAYER4, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    max_pool4 = max_pool_op(con2d4_2, 'max_pooling4')  # 7*7*128
    logits = tf.layers.conv2d(max_pool4, filters=MAX_BB_CNT * 5 * 16, kernel_size=1, strides=1, padding="same",
                              activation=None, name='fcn')
    # 7*7*2*5*16

    return logits


def backward():
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
        y_ = tf.placeholder(tf.float32, [None, GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])
        y = net(x)
        global_step = tf.Variable(0, trainable=False)

        processed_logits = process_logits(y)

        lambda_coords = tf.constant(5.0)
        lambda_noobj = tf.constant(0.5)

        yolo_loss = normalize_yolo_loss(y_, processed_logits, lambda_coords, lambda_noobj)
        loss = yolo_loss + tf.reduce_sum(tf.get_collection('losses'))

        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            5000,
            LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            if not os.path.exists(CHECK_POINT):
                os.mkdir(CHECK_POINT)

            tf.summary.merge_all()
            tf.summary.FileWriter(CHECK_POINT + "/summary", graph)

            batch_count = 0

            for i in range(STEPS):
                batch_x, batch_y = next_batch()
                batch_count += 1
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: batch_x, y_: batch_y})
                if i % 5 == 0:
                    print("STEP: {}, LOSS: {}.".format(step, loss_value))
                if i % 1000 == 0:
                    print("STEP: %-8d,BATCH: %-8d, LOSS: %-8g." % (step, batch_count, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def get_result():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
        y = process_logits(net(x))

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            # for test_image_id, _ in enumerate(test_ids):
            test_image_id = 1
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                result = sess.run([y], {x: np.reshape(test_images[test_image_id], [1, IMG_WIDTH, IMG_HEIGHT, 3])})
                boxes = result[0]
                boxes = np.reshape(boxes, newshape=[GRID_DIM, GRID_DIM, MAX_BB_CNT, 5])
                bbs = []

                for i in range(GRID_DIM):
                    for j in range(GRID_DIM):
                        for b in range(MAX_BB_CNT):
                            if (boxes[i][j][b][4] > 0.1):
                                grid_center_x = ((j + 0) * GRID_PIX + GRID_PIX / 2)
                                grid_center_y = ((i + 0) * GRID_PIX + GRID_PIX / 2)

                                new_box_center_x = boxes[i][j][b][0] * IMG_WIDTH + grid_center_x
                                new_box_center_y = boxes[i][j][b][1] * IMG_HEIGHT + grid_center_y

                                new_w = np.square(boxes[i][j][b][2]) * IMG_WIDTH
                                new_h = np.square(boxes[i][j][b][3]) * IMG_HEIGHT

                                x1 = new_box_center_x - new_w / 2
                                y1 = new_box_center_y - new_h / 2

                                x2 = new_box_center_x + new_w / 2
                                y2 = new_box_center_y + new_h / 2

                                bbs.append((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
                img = test_images[test_image_id]
                print(img)
                print(bbs)
                cv2.imshow('img',img)
                for i, b in enumerate(bbs):
                    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

                cv2.imshow('detect', img)
                cv2.waitKey(0)
            else:
                print('No checkpoint file found')
                return


if __name__ == '__main__':
    # re_build_size()
    # backward()
    re_build_size()
    get_result()