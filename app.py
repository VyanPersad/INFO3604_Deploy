from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.filters import gabor
import os
import cv2
import numpy as np
import math
import csv
import pandas as pd
import tensorflow.compat.v1 as tf 
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    cont = "" 
    diss = ""
    homo = ""
    ener = ""
    corr = ""
    asm = ""
    g1 = ""
    g2 = ""
    g3 = ""
    g4 = ""
    g5 = "" 
    
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        #file.save(os.path.join(app.config['UPLOAD'], filename))
        #img = os.path.join(app.config['UPLOAD'], filename)
        file.save(f'static/uploads/{filename}')
        img = f'static/uploads/{filename}'
        img = cv2.imread(img)
        lower = np.array([3, 15, 10], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        image = cv2.resize(img, (300, 300))
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(image, image, mask=skinMask)
        contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        non_black_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 100:
                non_black_boxes.append((x, y, w, h))
            
        skin_cropped = skin.copy()
        for box in non_black_boxes:
            x, y, w, h = box
            #img output,upper left, lower right, BGR Color, thickness
            cv2.rectangle(skin_cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_mask = np.zeros_like(skinMask)
        cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
        largest_contour_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)
        
        file_path = 'static/cropped.png'
        cv2.imwrite(file_path,largest_contour_image)
        
        img = cv2.imread(file_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #Calculate GLCM with specified parameters
        distances = [1]  # Distance between pixels
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for pixel pairs
        levels = 256  # Number of gray levels
        symmetric = True
        normed = True
        
        glcm = graycomatrix(gray_image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)        
      
        cont = round(graycoprops(glcm, 'contrast').ravel()[0], 4)
        diss = round(graycoprops(glcm, 'dissimilarity').ravel()[0], 4)
        homo = round(graycoprops(glcm, 'homogeneity').ravel()[0], 4)
        ener = round(graycoprops(glcm, 'energy').ravel()[0], 4)
        corr = round(graycoprops(glcm, 'correlation').ravel()[0], 4)
        asm = round(graycoprops(glcm, 'ASM').ravel()[0], 4)

        frequencies = [0.1, 0.3, 0.5]
        kernels = []

        # Generate Gabor filter kernels
        for frequency in frequencies:
            for theta in angles:
                kernel = np.real(gabor(gray_image, frequency, theta=theta))
                kernels.append(np.mean(kernel))

        # Convert the list of Gabor features to a numpy array
        gabor_features = np.array(kernels)
        g1 = round(gabor_features[0],4)
        g2 = round(gabor_features[1],4)
        g3 = round(gabor_features[2],4)
        g4 = round(gabor_features[3],4)
        g5 = round(gabor_features[4],4)
    
    return render_template('index.html', 
                                img_path1='cropped.png',
                                CONT=cont,DISS=diss, HOMO=homo,ENER=ener, CORR=corr,ASM=asm,
                                G1=g1,G2=g2,G3=g3,G4=g4,G5=g5,
                            )


#if __name__ == '__main__':
#    app.run() 