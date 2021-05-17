# -*- coding: utf-8 -*-
"""
"""
import streamlit as st
from PIL import Image, ImageEnhance
import cv2 
import os
import numpy as np
from mtcnn import MTCNN
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Thresholding','Contrast','Edge Detection', 'Watermark','Invisible Watermark','Face Detection')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Thresholding':
        thresholding()
    if selected_box == 'Edge Detection':
        edge_detection()
    if selected_box == 'Watermark':
        watermark()
    if selected_box == 'Face Detection':
        face_detection()
    if selected_box == 'Contrast':
        contrast()
    if selected_box == 'Invisible Watermark':
        invisible_watermarking()

def welcome():
    
    st.title('Digital Image Processing Project')
    
    st.subheader('A simple web app that shows different image processing algorithms. You can choose the options'
             + ' from the left. The algorithms have been implemented using the help of Streamlit. ')
    
    st.text('Done by COE17B008, COE17B011 and CDS21D0004')


def load_image(filename):
    image = cv2.imread(filename)
    return image
 
def thresholding():
    
    st.header("Thresholding")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        if st.button('See Original Image'):
            st.image(original, use_column_width=True)
        
        #img_array = np.array(original)
        #image = cv2.imread(img_array)

        file_bytes = np.array(original)
        #image = cv2.imdecode(file_bytes, 1)

        image = cv2.cvtColor(file_bytes, cv2.COLOR_BGR2GRAY)
        
        x = st.slider('Change Threshold value',min_value = 50,max_value = 255)  

        ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
        thresh1 = thresh1.astype(np.float64)
        st.image(thresh1, use_column_width=True,clamp = True)
        
        st.text("Bar Chart of the image")
        histr = cv2.calcHist([image],[0],None,[256],[0,256])
        st.bar_chart(histr)

def edge_detection():
    st.header("Edge detection using Canny Edge Detector")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        if st.button('See Original Image'):
            st.image(original, use_column_width=True)
        file_bytes = np.array(original)
        image = cv2.cvtColor(file_bytes, cv2.COLOR_BGR2GRAY)
        st.text("Press the button below to view Canny Edge Detection Technique")
        if st.button('Canny Edge Detector'):
            edges = cv2.Canny(file_bytes,50,300)
            cv2.imwrite('edges.jpg',edges)
            st.image(edges,use_column_width=True,clamp=True)    
         
def watermark():
    st.header("Watermarking")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_file2 = st.file_uploader("Upload the watermark image", type=["png", "jpg", "jpeg"])
        if uploaded_file2 is not None:
            original = Image.open(uploaded_file)
            watermark_img = Image.open(uploaded_file2)
            if st.button('See Original Image'):
                st.image(original, use_column_width=True)
            file_bytes = np.array(original)
            watermark_bytes=np.array(watermark_img)
            ext = os.path.splitext(uploaded_file.name)[-1]
            if (ext == ".png"):
                cv2.imwrite("1.jpg",file_bytes)
                file_bytes = cv2.imread("1.jpg")
            ext = os.path.splitext(uploaded_file2.name)[-1]
            if (ext == ".png"):
                cv2.imwrite("2.jpg",watermark_bytes)
                watermark_bytes = cv2.imread("2.jpg")
            oH,oW = file_bytes.shape[:2]
            file_bytes = np.dstack([file_bytes, np.ones((oH,oW), dtype="uint8") * 255])

            #resizing the watermark
            scl = 10
            w = int(watermark_bytes.shape[1] * scl / 100)
            h = int(watermark_bytes.shape[0] * scl / 100)
            dim = (w,h)
            lgo = cv2.resize(watermark_bytes, dim, interpolation = cv2.INTER_AREA)
            lH,lW = lgo.shape[:2]
            lgo = cv2.cvtColor(lgo,cv2.COLOR_RGB2RGBA)
            #blending
            ovr = np.zeros((oH,oW,4), dtype="uint8")
            ovr[oH - lH - 60:oH - 60, oW - lW - 10:oW - 10] = lgo
            final = file_bytes.copy()
            final = cv2.addWeighted(ovr,0.5,final,1.0,0,final)
            st.image(final,use_column_width=True,clamp=True)  


def face_detection(): 
    st.header("Face Detection")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        if st.button('See Original Image'):
            st.image(original, use_column_width=True)
        file_bytes = np.array(original)
        ext = os.path.splitext(uploaded_file.name)[-1]
        if (ext == ".png"):
            cv2.imwrite("3.jpg",file_bytes)
            file_bytes = cv2.imread("3.jpg")
        detector = MTCNN()
        faces=detector.detect_faces(file_bytes)
        for result in faces:
            # get coordinates
            x, y, width, height = result['box']
            # create the shape
            rect = cv2.rectangle(file_bytes,(x, y), (x+width, y+height), color=(255, 0, 0), thickness=2)
        st.text("The number of faces detected is: "+ str(len(faces)))
        st.image(file_bytes, use_column_width=True,clamp = True)

def invisible_watermarking():
    st.header("Invisible Watermarking")
    st.subheader("The invisible watermarking has been performed using DWT-DCT-SVD as a combination")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        if st.button('See Original Image'):
            st.image(original, use_column_width=True)
        file_bytes = np.array(original)
        wm = 'iniw'
        encoder = WatermarkEncoder()
        encoder.set_watermark('bytes', wm.encode('utf-8'))
        bgr_encoded = encoder.encode(file_bytes, 'dwtDctSvd')
        st.text("The watermarked image")
        st.image(bgr_encoded, use_column_width=True,clamp = True)
        
        decoder = WatermarkDecoder('bytes', 32)
        watermark = decoder.decode(bgr_encoded, 'dwtDctSvd')
        st.text("The decoded watermark is "+watermark.decode('utf-8'))

def contrast():
    st.header("Changing contrast")
    uploaded_file = st.file_uploader("Upload the image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        if st.button('See Original Image'):
            st.image(original, use_column_width=True)
        file_bytes = np.array(original)
        image = Image.fromarray(file_bytes)
        image = image.convert("RGB")
        x = st.slider('Change Contrast value',min_value = 0,max_value = 10)
        enhancer = ImageEnhance.Contrast(image)
        im_output = enhancer.enhance(x)
        st.image(im_output, use_column_width=True,clamp = True)

       
if __name__ == "__main__":
    main()