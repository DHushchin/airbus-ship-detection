import os
import sys
import path
import numpy as np

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from training.model import load_model
from constants import ROOT_PATH
from utils import evaluate, get_submission, save_submission

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    unet = load_model()
    evaluate(unet)
    test_paths = np.array(os.listdir(ROOT_PATH + '/data/test_v2'))
    sub = get_submission(test_paths, unet)
    save_submission(sub)
    
    
if __name__ == '__main__':
    main()    
    