import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import io
from tensorflow.keras.preprocessing import image
# Load the pre-trained model (adjust the model path as needed)
model = load_model(r'C:\Users\laksh\Python_Basics\CNN\Mood_Classification\model.h5')

# App title
st.title("Emotion Detection with CNN: Is the Person Happy or Not? 😃🤔")

# Upload an image
st.header('Upload an Image to Predict if the Person is Happy or Not!')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = image.load_img(uploaded_file, target_size=(200, 200))
    # Show the original image
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Adjust the quality of the image
    quality = st.slider('Select Image Quality (%)', 1, 100, 80)  # Quality slider from 1 to 100
    
    # Save the image to a BytesIO object with the selected quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Load the adjusted image from the buffer
    img_quality = Image.open(buffer)
    
    # Show the adjusted quality image
    st.image(img_quality, caption=f"Image with {quality}% Quality", use_column_width=True)

    
    # Preprocessing the image
    img_resized = img.resize((200, 200)) 
    x = image.img_to_array( img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Model prediction
    prediction = model.predict(x)
    
    # Displaying the result
    if prediction < 0.5:  # Adjust the threshold if needed
        st.write("🤗 **I am Happy!**")
        st.write('App is ready to predict happiness!')
    else:
        st.write("😞 **I am Not Happy!**")
        st.write('App is ready to predict UnHappiness!')

    
#===================================================================
st.write("========================================================================================")
st.title("Image Prediction App 📷")
st.markdown('''Select an image from the dropdown menu on the left sidebar.)
The app displays the chosen image and provides a prediction on whether the image conveys a "happy" or "not happy" emotion.''')
# Specify the directory containing images
img_dir = r"D:\pictures\testing"  # Update with your image directory path

# Sidebar: Instructions and image selection
st.sidebar.header("Instructions")
st.sidebar.write("1. Select an image from the dropdown.")
st.sidebar.write("2. The model will display the prediction.")

        

# Check if directory exists
if os.path.isdir(img_dir):
    # List all image files in the directory
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    # Sidebar: Dropdown to select an image
    selected_image = st.sidebar.selectbox("Select an image:", image_files)
    
    if selected_image:
        img_path = os.path.join(img_dir, selected_image)
        
        # Display selected image
        st.subheader("Selected Image From the sidebar")
        img = Image.open(img_path)
        st.image(img, caption=selected_image, use_column_width=True)
        
        # Preprocess the image for prediction
        img_resized = img.resize((200, 200))  # Resize to model's input size
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        
        # Interpret the prediction
        st.subheader("Prediction Result")
        if prediction[0][0] < 0.5:
            st.success("I am happy 😄")
        else:
            st.warning("I am not happy 😕")
        
else:
    st.sidebar.error(f"Directory '{img_dir}' not found.")
    st.error("Image directory not found. Please check the path.")

