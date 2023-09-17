import streamlit as st
from streamlit_option_menu import option_menu
import webbrowser
from app import *


def nav():
    # 1. as slidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title="SkeySpot YOLOV7 Object Detection",
            options=["YoloV7"],
            #menu_title= "AI Team 6",
            #options = ["YoloV7", "YoloV5"],
            #icons=['binoculars', 'binoculars-fill'],
        )
    url = ''

    if selected == "YoloV7":
        yolov7()


    st.sidebar.subheader('Service Keys Detectable')
    st.sidebar.write("Digital Electrical Layout Plans")
    #st.sidebar.write("Unripe strawberry")
    # st.sidebar.write("Duck")
    # st.sidebar.write("Chicken")
    # st.sidebar.write("Grapes")
    # st.sidebar.write("Watermelon")
