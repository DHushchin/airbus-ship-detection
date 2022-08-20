import tensorflow as tf
from keras.metrics import MeanIoU
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Activation, BatchNormalization
from keras import models
import sys
import path


directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from constants import ROOT_PATH
from training.utils import image_gen
                

def build_model(n_filters=32):
    
    input_layer = Input((None, None, 3))

    # Downsample
    conv1 = Conv2D(n_filters * 1, (3, 3), padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    convm = Conv2D(n_filters * 16, (3, 3), padding="same")(pool4)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)
    convm = Conv2D(n_filters * 16, (3, 3), activation="relu", padding="same")(convm)
    
    # Upsample
    deconv4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(n_filters * 8, (3, 3), padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Activation('relu')(uconv4)
    uconv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(n_filters * 4, (3, 3), padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(n_filters * 2, (3, 3), padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(n_filters * 1, (3, 3), padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model


def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / denominator


def load_model():
    unet = models.load_model(ROOT_PATH + '/model/model.h5', custom_objects={"dice_coef": dice_coef})
    return unet


def save_model(unet):
    unet.save_weights(ROOT_PATH + '/model/weights.h5')
    unet.save(ROOT_PATH + '/model/model.h5')
    

def train_model(train_df, valid_df):
    unet = build_model()
    print(unet.summary())

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.4, 
                                    patience=5, mode='min', cooldown=5)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=30)

    unet.compile(optimizer=Adam(), 
                loss=binary_crossentropy, 
                metrics=[MeanIoU(num_classes=2), dice_coef, 'binary_accuracy']) 
    
    history = [unet.fit(image_gen(train_df, 8),
                   steps_per_epoch=train_df.shape[0] // 1000,
                   epochs=1000,
                   validation_data=next(image_gen(valid_df, 8)),
                   shuffle=True,
                   callbacks=[early, reduceLROnPlat])]
    
    return unet, history
