import os
import pandas as pd

from model import train_model, ROOT_PATH
from utils import save_model, plot_history, process_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    masks = pd.read_csv(ROOT_PATH + '/data/train_ship_segmentations_v2.csv')
    train_df, valid_df = process_data(masks)
    unet, history = train_model(train_df, valid_df)
    save_model(unet)
    plot_history(history)
    
    
if __name__ == '__main__':
    main()
