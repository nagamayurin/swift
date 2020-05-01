import streamlit as st 
#import joblib,os
import spacy
nlp = spacy.load('en_core_web_sm')

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from summarizer import Summarizer
#import folium
import numpy as np
import nltk
from string import digits
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gtts import gTTS

import utils.display as udisp

import src.pages.home
import src.pages.summarytext
import src.pages.summaryurl

def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.sidebar.markdown(
        f"""
<style>
    .sidebar .sidebar-content, .sidebar .sidebar-toggle {{
        
        {max_width_str}
        background-image: url('file:///C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/summary_pageapp/pattern.png');
        padding-top: {0}rem;
        padding-right: {1}rem;
        padding-left: {1}rem;
        padding-bottom: {10}rem;
    }}
    .sidebar .sidebar-content, .sidebar .sidebar-toggle {{
        background-image: url('file:///C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/summary_pageapp/pattern.png');
        color: {'white'};
        background-color: #f4f1bb  ;
        text-align: justify;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
_set_block_container_style()

def _set_report_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        
        padding-top: {4}rem;
        padding-right: {4}rem;
        padding-left: {4}rem;
        padding-bottom: {10}rem;
    }}
    .reportview-container .main {{
        
        color: {'white'};
        background-color: {'black'};
        text-align: justify;
    }}
</style>
""",
        unsafe_allow_html=True,)
    
_set_report_container_style()

st.markdown("""
<style>
body {

    color: {'black'};
    background-color: {'white'}  ;
       
    
    etc. 
}
</style>

    
    """, unsafe_allow_html=True)



#st.sidebar.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:column;}.header{padding: 1px 16px; background: #fff; color: #fff; position:fixed;top:0;text-align: justify;} .sticky { position: fixed; top: 0; width: 100%;}</style>', unsafe_allow_html=True)

MENU = {
    "Home" : src.pages.home,
    "Summary for Text" : src.pages.summarytext,
    "Summary for URL" : src.pages.summaryurl       
}

#ADFF2F
#image2 = Image.open('logo.png')
#image3 = Image.open('logo2.png')
image4 = Image.open('logo_m2.png')

def main():
    st.markdown('<style>body { margin: 0; font-family: font-family: Tangerine;font-size:5px, Helvetica bold, sans-serif bold;font-size: 32px;text-align: justify} .header{padding: 15px 16px; background: black; color: #f4f1bb; position:fixed;top:0;text-align: center;} .sticky { position: fixed; top: 0; width: 100%;text-align: center; } </style><div class="header" id="myHeader"><b>'+str('Swift Summarator')+'</div>', unsafe_allow_html=True)
    #st.sidebar.image(image2, caption=None, width=170, use_column_width=False, clamp=True, channels='RGB')
    st.sidebar.image(image4, caption=None, width=250, use_column_width=False, clamp=True, channels='RGB')
    menu_selection = st.sidebar.radio("NAVIGATION", list(MENU.keys()))

    menu = MENU[menu_selection]

    with st.spinner(f"Loading {menu_selection} ..."):
        udisp.render_page(menu)

if __name__ == "__main__":
    main()
    

