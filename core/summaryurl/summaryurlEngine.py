import streamlit as st 
from datetime import datetime

import numpy as np
import utils.display as display
import utils.globalDefine as globalDefine
from urllib.request import Request, urlopen
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
from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#import pyttsx3
# initialisation 
#engine = pyttsx3.init() 
from gtts import gTTS
#from comtypes.client import CreateObject
#engine = CreateObject("SAPI.SpVoice")
#stream = CreateObject("SAPI.SpFileStream")
#image2 = Image.open('novartis_trans.png')
#image = Image.open('theme2.jpg')
#st.sidebar.image(image2, caption=None, width=150, use_column_width=False, clamp=True, channels='RGB')
#from spacy import displacy
#HTML_WRAPPER = """div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

#summary pkgs
#from gensim.summarization import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx,n):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,n)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

#Bi-grams
def ngrams(docx,n):
    #text = " ".join(docx)
    text1 = docx.lower()
    text2 = re.sub(r'[^a-zA-Z]'," ",text1)
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram    

def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

def analyze_text(text):
    return nlp(text)

#webscrapping pkgs
from bs4 import BeautifulSoup
from urllib.request import urlopen

@st.cache
def get_text(raw_url):
    #page = urlopen(raw_url)
    req = Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
    page = urlopen(req)
    #page = web_byte.decode('utf-8')
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

def sumurl_main(title, subtitle):
    #st.write("Summarize URL")    

    #st.sidebar.title(title)
    st.sidebar.info(
        subtitle
    )
#    st.info(
#                "Please paste your url into the left side box & select 'No of sentences' then click the 'Summarize!' to view the summary"
#                )

    #activities = ["Summarize","Summarize for URL","NER Checker","NER for URL"]
    #choice = st.radio("Select Activity",activities)
    #if choice == 'Summarize':
    #    st.info(
    #            "Please paste your text into the left side box & click the 'Summarize!' to view the summary"
    #            )

    #if choice == 'Summarize for URL':
    #    st.info(
    #            "Please paste your url into the left side box & click the 'Summarize!' to view the summary"
    #            )

    #st.sidebar.subheader("Summary from URL")
    raw_url = st.sidebar.text_input("Enter URL","Type here")
    value = st.sidebar.number_input("No of sentences", min_value=5, max_value=25)
        #text_length = st.sidebar.slider("Length to Preview",50,100)
#        text_length = st.slider("Length to Preview",50,100)
    if st.sidebar.button("Summarize!"):
        if raw_url != "Type here":
            result = get_text(raw_url)
            estimatedTime_org_url = readingTime(result)
            st.info("Original Reading time - {} mins".format(estimatedTime_org_url))
            summary_result_url = sumy_summarizer(result,value)
            st.write(summary_result_url)
            estimatedTime_res_url = readingTime(summary_result_url)
            st.info("Summary Reading time - {} mins".format(estimatedTime_res_url))
            tts = gTTS(text=summary_result_url, lang='en')
            tts.save("saved_file3.mp3")
            audio_file2 = open('saved_file3.mp3', 'rb')
            audio_bytes2 = audio_file2.read()
            st.audio(audio_bytes2, format='audio/mp3',start_time=0)
    #st.sidebar.subheader("Visualizations")
    visualize = ["Visualize","WordCloud","Bigrams","Trigrams"]
    choice2 = st.sidebar.selectbox("Visualization",visualize)
        #if choice2 == "Only Summary":
            
    if choice2 == "WordCloud":
        if raw_url != "Type here":
            result = get_text(raw_url)
            c_text = result
                #plt.figure(figsize=[70,50])
                #maskArray = np.array(Image.open("comment.png"))
            wordcloud = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='black',
                                  scale=3, relative_scaling = 0.5,random_state=1).generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
        
    if choice2 == "Bigrams":
        if raw_url != "Type here":
            result = get_text(raw_url)
            c_text=ngrams((result),2)
            for i in range(0,len(c_text)):
                c_text[i] = " ".join(c_text[i])
            Bigram_Freq_u = nltk.FreqDist(c_text)
            #maskArray = np.array(Image.open("comment.png"))
                    
                    #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
                    #plt.figure(figsize = (50,25))
            bigram_wordcloud_u = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='steelblue',
                                           scale=3, relative_scaling = 0.5, random_state=1).generate_from_frequencies(Bigram_Freq_u)
                    #plt.figure(figsize = (50,25))
            plt.imshow(bigram_wordcloud_u,interpolation = 'bilinear')
            plt.axis("off")
            st.pyplot()
            
    if choice2 == "Trigrams":
        if raw_url != "Type here":
            result = get_text(raw_url)
            c_text=ngrams((result),3)
            for i in range(0,len(c_text)):
                c_text[i] = " ".join(c_text[i])
            trigram_Freq_u = nltk.FreqDist(c_text)
                #maskArray = np.array(Image.open("comment.png"))
                
            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
            trigram_wordcloud_u = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='black',
                                            scale=3, relative_scaling = 0.5, random_state=1).generate_from_frequencies(trigram_Freq_u)
            #plt.figure(figsize = (50,25))
            plt.imshow(trigram_wordcloud_u,interpolation = 'bilinear')
            plt.axis("off")
            st.pyplot()
