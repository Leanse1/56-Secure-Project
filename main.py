import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('./bgs/bg.png')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/56_secure.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.markdown("<h2 style='color: white;'>{}</h2>".format(class_name), unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Score: {}%</h3>".format(int(conf_score * 1000) / 10), unsafe_allow_html=True)
