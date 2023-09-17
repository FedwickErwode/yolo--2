import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor
from PIL import Image
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
import os


def yolov7():


    st.header('SkeySpot Yolo V7 Tool')
    st.subheader('SkeySpot YOLO V7 Model Trained on Custom Dataset for Digital Electrical Layout Plans')
    # path to model
    path_model_file = "models/yolov7best.pt"

    source = ("Image Detection",
              #"Video Detection",
              "NULL"
              )
    options = st.selectbox("Select input", range(
        len(source)), format_func=lambda x: source[x])

    # Confidence
    confidence = st.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)
   # Draw thickness
    draw_thick = st.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=5
    )

    # read class.txt
    class_labels = ('BT Entry Point',
                    'Cat 6 Data Socke',
                    'Cat 6 Data Socket',
                    'Ceiling Mounted Continuous Extract Fan With Boost Mode Activated By Light Switch',
                    'Ceiling Mounted Continuous Extract Fan With Local Boost Switch',
                    'Ceiling Mounted Continuous Extract FanWith Boost Mode Activated By Light Switch',
                    'Ceiling Mounted Continuous Extract FanWith Local Boost Switch',
                    'Co-Ax TV Socket',
                    'Consumer Unit',
                    'D',
                    'Double Socket',
                    'Double SocketCeiling Mounted Continuous Extract Fan With Boost Mode Activated By Light Switch',
                    'Electric Meter Box',
                    'External Wall Light',
                    'Full Height Tiling',
                    'Fused spur',
                    'Gas Meter Box',
                    'Grid Switch',
                    'Hob Switch',
                    'Internal Wall Light',
                    'Light Switch',
                    'Low Energy Downlighter',
                    'Low Energy Pendant Light',
                    'Mains Wired Smoke Detector',
                    'Outside Socket',
                    'Outside Tap',
                    'Oven Switch',
                    'Programmable Room Thermostat',
                    'Radiator',
                    'Recirculating Extractor Fan',
                    'Shaver Socket',
                    'Single Socket',
                    'TV - Satellite Multisocket',
                    'Telephone Socket',
                    'Telephone SocketUnderfloor Heating Manifold',
                    'Track Light',
                    'Twin LED Strip Light',
                    'USB Double Socket',
                    'Underfloor Heating Manifold',
                    'Water Entry Position')

    # for i in range(len(class_labels)):
    #         classname = class_labels[i]

    # Image

    # def ImageInput():
    #     image_file = st.file_uploader(
    #         'Upload Image', type=['jpg', 'jpeg', 'png'])
    #     col1, col2 = st.columns(2)
    #     if image_file is not None:

    #         file_bytes = np.asarray(
    #             bytearray(image_file.read()), dtype=np.uint8)
    #         img = cv2.imdecode(file_bytes, 1)
    #         imga = Image.open(image_file)
    #         with col1:
    #             st.image(imga, caption='Uploaded Image',
    #                      use_column_width='always')
    #         #FRAME_WINDOW.image(img, channels='BGR')

    #         model = custom(path_or_model=path_model_file)

    #         bbox_list = []
    #         current_no_class = []
    #         pred = model(img)
    #         box = pred.pandas().xyxy[0]
    #         class_list = box['class'].to_list()

    #         for i in box.index:
    #             xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
    #                 int(box['ymax'][i]), box['confidence'][i]
    #             if conf > confidence:
    #                 bbox_list.append([xmin, ymin, xmax, ymax])
    #         if len(bbox_list) != 0:
    #             for bbox, id in zip(bbox_list, class_list):
    #                 plot_one_box(
    #                     bbox, img, label=class_labels[id], line_thickness=draw_thick)
    #                 current_no_class.append([class_labels[id]])
    #         with col2:
    #             st.image(img, channels='BGR', caption='Model Prediction(s)')

    #         # Current number of classes
    #         class_fq = dict(
    #             Counter(i for sub in current_no_class for i in set(sub)))
    #         class_fq = json.dumps(class_fq, indent=4)
    #         class_fq = json.loads(class_fq)
    #         df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

    #         # Updating Inference results
    #         with st.container():
    #             st.markdown("<h2>Inference Statistics</h2>",
    #                         unsafe_allow_html=True)
    #             st.markdown(
    #                 "<h3>Detected objects in current Frame</h3>", unsafe_allow_html=True)
    #             st.dataframe(df_fq, use_container_width=True)

    def ImageInput():
        uploaded_images = st.file_uploader(
            'Upload Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if uploaded_images:
            col1, col2 = st.columns(2)

            for idx, image_file in enumerate(uploaded_images):
                with col1:
                    st.image(image_file, caption=f'Uploaded Image {idx+1}',
                             use_column_width='always')

                file_bytes = np.asarray(
                    bytearray(image_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                imga = Image.open(image_file)

                # Perform prediction for the current image
                model = custom(path_or_model=path_model_file)
                bbox_list = []
                current_no_class = []
                pred = model(img)
                box = pred.pandas().xyxy[0]
                class_list = box['class'].to_list()

                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])

                if len(bbox_list) != 0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(
                            bbox, img, label=class_labels[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])

                with col2:
                    st.image(img, channels='BGR',
                             caption=f'Model Prediction(s) for Image {idx+1}')

                # Current number of classes
                class_fq = dict(
                    Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent=4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=[
                                     'Class', 'Number'])

                # Updating Inference results
                with st.container():
                    st.markdown(f"<h2>Inference Statistics for Image {idx+1}</h2>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<h3>Detected objects in Image {idx+1}</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq, use_container_width=True)


    if options == 0:
        ImageInput()
    # elif options == 1:
    #     videoInput()

