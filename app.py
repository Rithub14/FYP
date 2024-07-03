import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time


def imageInput(device, src):
    
    
    ### Tree Detection ###
    if src == 'Tree Detection':
        st.subheader('Tree Detection')
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads/tree/', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs/tree/', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # Call Model prediction
           
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/tree/best.pt', force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction', use_column_width='always')

    ### Variety Identification ###
    elif src == 'Variety Identification':
        st.subheader('Variety Identification')
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads/variety/', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs/variety/', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # Call Model prediction
           
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/variety/best.pt', force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction', use_column_width='always')


    ### Disease Detection ##
    elif src == 'Disease Detection':
        st.subheader('Disease Detection')
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads/disease/', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs/disease/', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # Call Model prediction
           
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/disease/best.pt', force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # Display predicton 

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction', use_column_width='always')



# Main
def main():
    # Sidebar
    
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Tree Detection', 'Variety Identification', 'Disease Detection'])

    # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=0)
    # End of Sidebar

    st.header('🥭 Mango')
    st.subheader('👈 Select from the options')
    

    imageInput(deviceoption, datasrc)

    # Hide Default Menu and Streamlit tag at the end
    #visibility: hidden or visible
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;} 
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()


