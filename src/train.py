import pandas as pd
from keras.metrics import MeanIoU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
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

train_gen = make_image_gen(train_df)
val_gen = make_image_gen(valid_df)

train_x, train_y = next(train_gen)
valid_x, valid_y = next(val_gen)

test_gen = aug_gen(train_gen)
test_x, test_y = next(test_gen)

unet = build_model(train_x.shape[1:])
print(unet.summary())

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, 
                                   mode='min', cooldown=3, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=50)

unet.compile(optimizer=Adam(), 
             loss=dice_coef, 
             metrics=[MeanIoU(num_classes=2)])                                                                

loss_history = unet.fit(aug_gen(make_image_gen(train_df)),
                        steps_per_epoch=train_df.shape[0] // 1000,
                        epochs=1000,
                        validation_data=next(make_image_gen(valid_df)),
                        callbacks=[early, reduceLROnPlat],)

unet.save_weights(ROOT_PATH + '/model/weights.h5')
unet.save(ROOT_PATH + '/model/model.h5')
