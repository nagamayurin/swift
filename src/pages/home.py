import pathlib
import utils.display as udisp

import streamlit as st

from PIL import Image

image = Image.open('theme2.jpg')
#st.sidebar.image(image2, caption=None, width=170, use_column_width=False, clamp=True, channels='RGB')


def write():
    
    st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels='RGB')
    #st.title("Automated Summary App")
    udisp.render_md("resources/home_info.md")
