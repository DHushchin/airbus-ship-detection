import pandas as pd
from keras.metrics import MeanIoU
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
masks = pd.read_csv(ROOT_PATH + '/data/train_ship_segmentations_v2.csv')
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
masks.drop(['ships'], axis=1, inplace=True)

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(1000) if len(x) > 1000 else x)

train_ids, valid_ids = train_test_split(balanced_train_df, 
                                        test_size = 0.2, 
                                        stratify = balanced_train_df['ships'])

train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)       

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

unet.save_weights(ROOT_PATH + '/model/weights.h5')
unet.save(ROOT_PATH + '/model/model.h5')

epochs = np.concatenate([mh.epoch for mh in history])
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].plot(epochs, np.concatenate([mh.history['loss'] for mh in history]), 'b-',
             epochs, np.concatenate([mh.history['val_loss'] for mh in history]), 'r-')
axes[0].legend(['Training', 'Validation'])
axes[0].set_title('Loss')


axes[1].plot(epochs, np.concatenate([mh.history['dice_coef'] for mh in history]), 'b-',
             epochs, np.concatenate([mh.history['val_dice_coef'] for mh in history]), 'r-')
axes[1].legend(['Training', 'Validation'])
axes[1].set_title('Dice Coefficient')

plt.show()
