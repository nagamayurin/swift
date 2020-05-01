import pathlib
import utils.display as udisp

import streamlit as st

#import core.summarytext.summarytextEngine as summarytexteng
import summarytextEngine as summarytexteng

def write():
    #udisp.title_awesome("Summarized Text")
    #st.title("Summarize Text")
    summarytexteng.sum_main("Summarize Text", "Enter or paste text in the below box")
    udisp.render_md("resources/summarytext2.md")
    udisp.render_md("resources/summarytext.md")
    
#    if st.checkbox("Show help document"):
#        st.code(udisp.render_md("resources/summarytext.md"))

    

    #udisp.render_md("resources/summarytext.md")
    #st.write("@nagamayuri")