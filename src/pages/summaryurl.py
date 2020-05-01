import pathlib
import utils.display as udisp

import streamlit as st
import summaryurlEngine as summaryurleng

def write():
    #st.title("Summarize URL")
    summaryurleng.sumurl_main("Summarize URL", "Enter or paste url in the below box")
    udisp.render_md("resources/summaryurl2.md")
    udisp.render_md("resources/summaryurl.md")
#    if st.checkbox("Show help document"):
#        st.code(udisp.render_md("resources/summaryurl.md"))
    