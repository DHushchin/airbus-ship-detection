import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import pathlib

directory = pathlib.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from img_processing import masks_as_image, imread
from constants import ROOT_PATH, BATCH_SIZE


def image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = ROOT_PATH + f'/data/train_v2/{c_img_id}'
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            
            c_img = c_img[::3, ::3]
            c_mask = c_mask[::3, ::3]
            
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
              
                
def process_data(masks):
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    masks.drop(['ships'], axis=1, inplace=True)

    balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(1000) if len(x) > 1000 else x)

    train_ids, valid_ids = train_test_split(balanced_train_df, 
                                            test_size = 0.2, 
                                            stratify = balanced_train_df['ships'])

    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids) 
    return train_df, valid_df

    
def plot_history(history):
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