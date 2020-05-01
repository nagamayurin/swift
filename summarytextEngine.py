import streamlit as st 
from datetime import datetime

import numpy as np

import utils.display as display
import utils.globalDefine as globalDefine

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
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#import pyttsx3
# initialisation 
#engine = pyttsx3.init() 
from gtts import gTTS

#image2 = Image.open('novartis_trans.png')

#st.sidebar.image(image2, caption=None, width=150, use_column_width=False, clamp=True, channels='RGB')

from spacy import displacy
HTML_WRAPPER = """div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

#summary pkgs
from gensim.summarization import summarize

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
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

def sum_main(title, subtitle):
    #st.write("Summarize Text")    

    #st.sidebar.title(title)
    st.sidebar.info(
        subtitle
    )
#    st.info(
#                "Please paste your text into the left side box & select 'No of sentences' then click the 'Summarize!' to view the summary"
#                )
    #st.sidebar.title("About")
    #if st.sidebar.button("About this app"):
    #    st.sidebar.info(
    #            "This is an auto summarizer app for text articles, extracting the most important sentences by using NLP algorithms. It helps us to save time in our busy schedules who prefer to read the summary of those articles before we decide to jump in for reading entire article."
    #            )
    
    #st.write('<style>body { margin: 0; font-family: font-family: Tangerine;font-size:48px, Helvetica, sans-serif;font-size: 30px;text-align: center;} .header{padding: 10px 16px; background: #eaf4ff; color: #111; position:fixed;top:0;text-align: center;} .sticky { position: center; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str('RESUNER')+'</div>', unsafe_allow_html=True)
    #st.write('<style>body { margin: 0; font-family: font-family: Tangerine;font-size:48px, Helvetica, sans-serif;font-size: 30px;text-align: justify;} .header{padding: 10px 16px; background: #eaf4ff; color: #111; position:fixed;top:0;text-align: center;} .sticky { position: fixed; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str('Summary Generator and Entity Recognizer')+'</div>', unsafe_allow_html=True)
    #st.title("Summary Generator and Entity checker")
    #activities = ["Summarize","Summarize for URL","NER Checker","NER for URL"]
    #choice = st.radio("Select Activity",activities)
#    if choice == 'Summarize':
#        st.info(
#                "Please paste your text into the left side box & click the 'Summarize!' to view the summary"
#                )
#        st.sidebar.subheader("Summarization")
    raw_text = st.sidebar.text_area("Enter Text Here")
    value = st.sidebar.number_input("No of sentences", min_value=5, max_value=25)
            
        #summary_choice = st.selectbox("Summary Choice",["Gensim","Sumy Lex Rank"])
    if st.sidebar.button("Summarize!"):
        summary_result = sumy_summarizer(raw_text,value)
        estimatedTime_org = readingTime(raw_text)
            #text_length = st.slider("Length to Preview",50,100)
        st.info("Original Reading time - {} mins".format(estimatedTime_org))

        st.write(summary_result)
        estimatedTime_res = readingTime(summary_result)
        st.info("Summary Reading time - {} mins".format(estimatedTime_res))
            
            #engine = pyttsx3.init(driverName='sapi5')
            #infile = "tanjil.txt"
           # f = open(infile, 'r')
            #theText = f.read()
            #f.close()

            #Saving part starts from here 
        tts = gTTS(text=summary_result, lang='en')
            #saved_file=talkToMe(summary_result , lgg ='en')  
        tts.save("saved_file.mp3")
        audio_file = open('saved_file.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3',start_time=0)
    #st.sidebar.subheader("Visualizations")
    visualize = ["Visualize","WordCloud","Bigrams","Trigrams"]
    choice2 = st.sidebar.selectbox("Visualization",visualize)
        #if choice2 == "Only Summary":
            
    if choice2 == "WordCloud":
        c_text = raw_text
            #plt.figure(figsize=[70,50])
            #maskArray = np.array(Image.open("comment.png"))
        wordcloud = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='black',
                              scale=3, relative_scaling = 0.5, random_state=1).generate(c_text)
        plt.imshow(wordcloud,interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
            
    if choice2 == "Bigrams":
        c_text=ngrams((raw_text),2)
        for i in range(0,len(c_text)):
            c_text[i] = " ".join(c_text[i])
        Bigram_Freq = nltk.FreqDist(c_text)
            #maskArray = np.array(Image.open("comment.png"))

            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
        bigram_wordcloud = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='black',
                                     scale=3, relative_scaling = 0.5, random_state=1).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
        plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
        plt.axis("off")
#            maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/cloud2.png"))
    #wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwords_wc, background_color='white', mask = maskArray).generate_from_frequencies(dict(words_freq))
#            wordCloud = WordCloud(max_font_size=150,max_words=2000, margin=10, background_color='white', mask = maskArray,
#                                  scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(c_text)
#            plt.title('Most frequently occurring bigrams connected by same colour and font size')
#            plt.imshow(wordCloud, interpolation='bilinear')
#            plt.axis("off")
            #return st.pyplot()
        st.pyplot()
            
    if choice2 == "Trigrams":
        c_text=ngrams((raw_text),3)
        for i in range(0,len(c_text)):
            c_text[i] = " ".join(c_text[i])
        trigram_Freq = nltk.FreqDist(c_text)
            #maskArray = np.array(Image.open("comment.png"))
            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
        trigram_wordcloud = WordCloud(max_font_size=50,max_words=100, margin=10, background_color='white', contour_width=3,contour_color='black',
                                      scale=3, relative_scaling = 0.5, random_state=1).generate_from_frequencies(trigram_Freq)
            #plt.figure(figsize = (50,25))
        plt.imshow(trigram_wordcloud,interpolation = 'bilinear')
        plt.axis("off")
        st.pyplot()


#if __name__ == '__main__':
#    sum_main()
