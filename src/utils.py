import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
BATCH_SIZE = 4


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def make_image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = ROOT_PATH + '/data/train_v2/{}'.format(c_img_id)
            c_img = plt.imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            
            c_img = c_img[::3, ::3]
            c_mask = c_mask[::3, ::3]
            
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
 
def aug_gen(img_gen):
    aug_gen = ImageDataGenerator(shear_range=0.5, 
                                rotation_range=50, 
                                zoom_range=0.2, 
                                width_shift_range=0.2, 
                                height_shift_range=0.2, 
                                fill_mode='reflect')
    for x, y in img_gen:
        seed = np.random.choice(range(9999))
        res_x = aug_gen.flow(x, 
                             batch_size = x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        
        res_y = aug_gen.flow(y, 
                             batch_size = x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(res_x), next(res_y)
         
                
def build_model(input_shape, n_filters=32):
    
    input_layer = Input(input_shape)

    # Downsample
    conv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    convm = Conv2D(n_filters * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Dropout(0.5)(convm)
    convm = Conv2D(n_filters * 16, (3, 3), activation="relu", padding="same")(convm)
    
    # Upsample
    deconv4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator
