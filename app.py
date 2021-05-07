import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer #streamlit-webrtc helps to deal with 
                                                                   #real-time video streams.
import tensorflow as tf
from tensorflow import keras 



my_model = tf.keras.models.load_model('model.h5')
# loading .h5 file of our model and storing it in my_model

def draw_border(img, pt1, pt2, color, thickness, r, d):             # this function is just for the decorative square around the face
        x1,y1 = pt1                                                 # by using this, we will have 4 round corners around the face, instead of rectangle
        x2,y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)



class VideoTransformer(VideoTransformerBase):
    
    

    def transform(self, frame): 
    #transform() method, which transforms each frame coming from the video stream.
        img = frame.to_ndarray(format="bgr24")                                      # coverting captured image into array of pixels
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # load cascade classifier
        
        
        class_labels = ['Fear','Angry','Neutral','Happy']                 # the prediction will be number from 0-3 ; 
                                                                          #to link it to its emotion we created this list.


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  # converting image into grayscale
        face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)          # ROI (region of interest of detected face, stored as tuple of bottom left
        if face_roi is ():                                                # check if face_roi is empty ie. no face detected
            return img

        for(x,y,w,h) in face_roi:                                         # iterate through faces and draw rectangle over each face
            x = x - 5
            w = w + 10
            y = y + 7
            h = h + 2
            draw_border(frame, (x,y),(x+w,y+h),(0,0,204), 2,15, 10)       ## drawing some fancy border     
            img_color_crop = img[y:y+h,x:x+w]                             # croping colour image
            final_image = cv2.resize(img_color_crop, (48,48))             # size of colured image is resized to 224,224
            final_image = np.expand_dims(final_image, axis = 0)           # array is expanded by inserting axis at position 0
            final_image = final_image/255.0                               # feature scaling of final image
            prediction = my_model.predict(final_image)                    # emotion of the captured image is detected with the help of our model
            label=class_labels[prediction.argmax()]                       # we find the label of class which has maximaum probalility 
            cv2.putText(img,label, (50,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (120,10,200),3)    
                                                                          # putText is used to draw a detected label on image
                                                                          # (50,60)-top left coordinate   FONT_HERSHEY_SCRIPT_COMPLEX-font type
                                                                          # 2-fontscale   (120,10,200)-font colour   3-font thickness
                           
        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    # image captured from webcam is sent to VideoTransformer function
#webrtc_streamer can take video_transformer_factory argument, which is a Callable that returns an instance of a class which has transform(self, frame) method.
       
         

