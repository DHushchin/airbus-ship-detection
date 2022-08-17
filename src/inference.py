import pandas as pd
from utils import *

test_paths = np.array(os.listdir(ROOT_PATH + '/data/test_v2'))

sub = []

for c_img_name in test_paths:
    sub += pred_encode(c_img_name)
    
sub_df = pd.DataFrame(sub)
sub_df.columns = ['ImageId', 'EncodedPixels']
sub_df = sub_df[sub_df.EncodedPixels.notnull()]

sub_df.to_csv(ROOT_PATH + '/data/submission.csv', index=False)