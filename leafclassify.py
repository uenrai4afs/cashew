import streamlit as st
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import *
from keras import preprocessing
import time
import base64
#added by patrick
from cv2 import *
## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##


# fig = plt.figure()
st.title(':white[AI4AFS-UENR]')
st.header(':white[Cashew Disease/Pest Detection App]')

#st.markdown("Prediction Platform")
def set_background(main_bg):  # local image
    # set bg name
    main_bg_ext = "png"
    st.markdown(
        f"""
             <style>
             .stApp {{
                 background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                 background-size: cover
             }}
             </style>
             """,
        unsafe_allow_html=True
    )


set_background('cashew.png')
def main():
    ##====Juat upload image from gallery========
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=False)
    class_btn = st.button("Detect")
    camera_btn = st.button("Camera")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('Results')
                st.write(predictions)
        if camera_btn:
            webcam()
                    
    #====Added by Patrick for Camera=============
    
    
    


    
## -----------------------------------------------------x---------------------------------------x--------------------------------------------##
def webcam():
    # program to capture single image from webcam in python 
    # importing OpenCV library 
    #from cv2 import *
    cam_port = 0
    cam = VideoCapture(cam_port) 
    result, image = cam.read() 
    # If image will detected without any error, 
    # show result 
    if result: 

	    # showing result, it take frame name and image 
	    # output 
	    imshow("GeeksForGeeks", image) 

	    # saving image in local storage 
	    imwrite("GeeksForGeeks.png", image) 

	    # If keyboard interrupt occurs, destroy image 
	    # window 
	    waitKey(0) 
	    destroyWindow("GeeksForGeeks") 

    # If captured image is corrupted, moving to else part 
    else: 
	    print("No image detected. Please! try again") 
#=======================Above by patrick==================================================

## ============this code for format tflite file=========================
def predict(image):
    #model = "leaves_model.tflite"
    model="cashew.tflite"
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = np.array(image.resize((224, 224)), dtype=np.float32)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])

    labels = {0: "anthracnose", 1: "gumosis", 2: "healthy", 3: "leaf miner", 4: "non type", 5:"red rust" }
    label_new=["anthracnose", "gumosis", "healthy", "leaf miner", "non type", "red rust" ]


    label_to_probabilities = []
    print(probabilities)

    for i, probability in enumerate(probabilities):
        label_to_probabilities.append([labels[i], float(probability)])

    sorted(label_to_probabilities, key=lambda element: element[1])

    #result = {'healthy': 0, 'diseased': 0}
    #result = {'leaf Blight': 0, 'brown spot': 0, 'greenmite': 0, 'healthy': 0, 'mosaic': 0}
    #result = f"{label_to_probabilities[np.argmax(probability)][0]} with a {(100 * np.max(probabilities)).round(2)} % confidence."
    #result=f"{} with a {}"
    high=np.argmax(probabilities)
    result_1=label_new[high]
    confidence=100 * np.max(probabilities)
    result="Category:"+ "  "+str(result_1) +"     "+ "\n Confidence: "+ " "+ str(confidence)+ "%"

    return result


if __name__ == "__main__":
    main()
