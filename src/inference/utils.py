import sys
import path

import pandas as pd
import numpy as np

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from img_processing import encode_predict
from training.utils import process_data, image_gen
from constants import ROOT_PATH, BATCH_SIZE

def get_submission(test_paths, unet):
    sub = []
    for img_id in test_paths:
        sub += encode_predict(img_id, unet)
        print(sub)
    return sub


def save_submission(sub):
    sub_df = pd.DataFrame(sub)
    sub_df.columns = ['ImageId', 'EncodedPixels']
    sub_df = sub_df[sub_df.EncodedPixels.notnull()]

    sub_df.to_csv(ROOT_PATH + '/data/submission.csv', index=False)
    

def evaluate(unet):
    masks = pd.read_csv(ROOT_PATH + '/data/train_ship_segmentations_v2.csv')
    train_df, valid_df = process_data(masks)
    unet.evaluate(image_gen(valid_df), steps=np.ceil(valid_df.shape[0] / BATCH_SIZE))