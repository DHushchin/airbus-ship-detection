import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from utils import *
from skimage import exposure

st.set_page_config("Airbus Ships")
st.header("Airbus ship image segmentation")

def segment_image(img):
    cur_seg, _ = raw_prediction(img)
    img = cur_seg[:,:,0]
    arr_rescaled = exposure.rescale_intensity(img)
    return arr_rescaled


def main():
    col1, col2 = st.columns(2)
    try:
        file_uploaded = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if file_uploaded is not None:
            st.balloons()
            image = Image.open(file_uploaded)
            image = image.resize((768, 768))
            with st.spinner("Loading image..."):
                col1.header("Uploaded image")
                col1.image(image, use_column_width=True)
                col2.header("Image segmentation")
                img = imread(file_uploaded)
                col2.image(segment_image(img), use_column_width=True)
    except:
        st.error('Ooops, something went wrong')
        
        
if __name__ == '__main__':
    main()
            