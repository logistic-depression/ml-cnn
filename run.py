"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf

from tensorflow.keras import backend as K
import gc

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2

TOTAL_DATA_SIZE=100
DATA_IDS = numpy.array([i for i in range(1,TOTAL_DATA_SIZE+1) if i != 33])
TRAIN_SIZE = 89
VALIDATE_SIZE = 10
ROTATION = True

numpy.random.seed(42)
IDS = numpy.random.choice(DATA_IDS, size=(TRAIN_SIZE+VALIDATE_SIZE), replace=False, p=None)
TRAIN_IDS = IDS[:TRAIN_SIZE]
VALIDATE_IDS = IDS[TRAIN_SIZE:]
if ROTATION:
    TRAIN_IDS = numpy.array([j for i in TRAIN_IDS for j in range(i, TOTAL_DATA_SIZE*8+1, TOTAL_DATA_SIZE) ])
    VALIDATE_IDS = numpy.array([j for i in VALIDATE_IDS for j in range(i, TOTAL_DATA_SIZE*8+1, TOTAL_DATA_SIZE) ])


GROUPED_BATCH_SIZE = 3

#VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

tf.app.flags.DEFINE_string('train_dir', '/tmp/segment_aerial_images',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

TRAIN_IDS.shape, VALIDATE_IDS.shape, TRAIN_IDS[:3], VALIDATE_IDS[:3]

# Extract patches from a given image
def img_crop(im, w, h):
    h_shift = (GROUPED_BATCH_SIZE // 2)*h
    w_shift = (GROUPED_BATCH_SIZE // 2)*w
    is_2d = len(im.shape) < 3
    if is_2d:
        im = numpy.pad(im, ((w_shift, w_shift),(h_shift, h_shift)), 'symmetric')
    else:
        im = numpy.pad(im, ((w_shift, w_shift),(h_shift, h_shift),(0,0)), 'symmetric')
    gc.collect()
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(h_shift, imgheight - h_shift, h):
        for j in range(w_shift, imgwidth - w_shift, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[
                    j-w_shift:j+w+w_shift,
                    i-h_shift:i+h+h_shift,
                    :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, image_ids):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in image_ids:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        #return [0, 1]
        return 1
    else:  # bgrd
        #return [1, 0]
        return 0

# Extract label images
def extract_labels(filename, image_ids):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in image_ids:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def mcor(y_true, y_pred):
    #matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + K.epsilon())

def bcor(y_true, y_pred):
    pp = K.mean(K.round(K.clip(y_pred, 0, 1)))
    pn = 1 - pp
    pos = K.mean(K.round(K.clip(y_true, 0, 1)))
    neg = 1 - pos
    
    tp = K.mean(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = pp - tp
    
    fn = pos - tp
    tn = pn - fn
    
    return (tp - (pp*pos)) / (pos - (pos*pos))

def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pp + K.epsilon())
    return precision

def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (pos + K.epsilon())
    return recall

def f1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pp + K.epsilon())
    
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (pos + K.epsilon())
    
    return 2*((precision * recall) / (precision + recall + K.epsilon()))

data_dir = 'training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
test_data_filename = 'test_set_images/'

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, TRAIN_IDS)
validate_data = extract_data(train_data_filename, VALIDATE_IDS)

train_labels = extract_labels(train_labels_filename, TRAIN_IDS)
validate_labels = extract_labels(train_labels_filename, VALIDATE_IDS)

#tn = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED)
#const = tf.keras.initializers.Constant(value=0.1)
model = tf.keras.models.Sequential([ # 48
  tf.keras.layers.Conv2D(32, (5, 5), padding="same",
                         activation=tf.nn.relu
                        ), # 48
  tf.keras.layers.MaxPooling2D(), # 24
  tf.keras.layers.Conv2D(64, (5, 5), padding="same",
                         activation=tf.nn.relu
                        ), # 24
  tf.keras.layers.MaxPooling2D(), # 12
  tf.keras.layers.Dropout(rate=0.25),
  tf.keras.layers.Conv2D(128, (3, 3), padding="same",
                         activation=tf.nn.relu
                        ), # 12
  tf.keras.layers.MaxPooling2D(), # 6
  tf.keras.layers.Conv2D(256, (3, 3), padding="same",
                         activation=tf.nn.relu
                        ), # 6
  tf.keras.layers.MaxPooling2D(), # 3
  tf.keras.layers.Dropout(rate=0.25),
  tf.keras.layers.Conv2D(256, (3, 3),
                         activation=tf.nn.relu
                        ), # 1
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512,
                         activation=tf.nn.relu
                       ),
  tf.keras.layers.Dense(1,
                         activation=tf.nn.sigmoid
                       ),
])

gc.collect()

# Compile model
optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0002, amsgrad=False)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', f1, precision, recall, mcor, bcor])

RESTORE_MODEL

# Train
if not RESTORE_MODEL:
    model.fit(train_data, train_labels, batch_size=256, epochs=10)
    model.save_weights("./weights.hdf5")
else:
    model.load_weights("./weights.hdf5")

model.summary()

model.evaluate(validate_data, validate_labels)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = -l
            idx = idx + 1
    return array_labels

# Get prediction for given input image
def get_prediction(img, size=400):
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    pred = model.predict(data)
    img_prediction = label_to_img(size, size, IMG_PATCH_SIZE, IMG_PATCH_SIZE, pred)
    
    return img_prediction

# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(filename, image_idx):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    cimg = concatenate_images(img, img_prediction)

    return cimg

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

print("Running prediction on validate set")
prediction_training_dir = "predictions_training/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in VALIDATE_IDS:
    pimg = get_prediction_with_groundtruth(train_data_filename, i)
    Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
    oimg = get_prediction_with_overlay(train_data_filename, i)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

print("Running prediction on test set")
prediction_test_dir = "prediction_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1,51):
    imageid = "test_" + str(i)
    image_filename = test_data_filename + imageid + "/" +  imageid + ".png"
    img = mpimg.imread(image_filename)
    
    pimg = get_prediction(img, size=608)
    
    Image.fromarray(numpy.uint8((1+pimg)*255.0)).save(prediction_test_dir + "prediction_" + str(i) + ".png")

