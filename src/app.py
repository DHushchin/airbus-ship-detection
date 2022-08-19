import streamlit as st
from skimage import exposure
from PIL import Image

from img_processing import image_predict, imread
from train.model import load_model


def segment_image(img, model):
    cur_seg, _ = image_predict(img, model)
    img = cur_seg[:,:,0]
    arr_rescaled = exposure.rescale_intensity(img)
    return arr_rescaled


def main():
    st.set_page_config("Airbus Ships")
    st.header("Airbus ship image segmentation")
    col1, col2 = st.columns(2)
    unet = load_model()
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
                col2.image(segment_image(img, unet), use_column_width=True)
    except Exception as e:
        st.error('Ooops, something went wrong', e)
        
        
if __name__ == '__main__':
    main()
            