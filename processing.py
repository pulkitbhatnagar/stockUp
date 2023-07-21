#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:24:16 2023

@author: pb185199
"""

# importing the required library  
from imageai.Detection import ObjectDetection  

from typing import Union

from fastapi import FastAPI

app = FastAPI()
  
# instantiating the class  
recognizer = ObjectDetection()  
#creating dictnary

result={}
  
# defining the paths  
path_model = "./Models/yolo-tiny.h5"  
path_input = "./Input/image4.jpeg"  
path_output = "./Output/newimage3.jpg"  
  
# using the setModelTypeAsTinyYOLOv3() function  
recognizer.setModelTypeAsTinyYOLOv3()  
# setting the path of the Model  
recognizer.setModelPath(path_model)  
# loading the model  
recognizer.loadModel()  
# calling the detectObjectsFromImage() function  
recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  





# iterating through the items found in the image  
for eachItem in recognition:  
    result[eachItem["name"]]=eachItem["percentage_probability"]
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  


@app.get("/")
def read_root():
    return result
