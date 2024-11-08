# Emotion-Detection-Using-CNN
**Objective:**
To build and deploy an emotion detection system using a Convolutional Neural Network (CNN) that classifies a person’s facial expression as **"Happy" or "Not Happy."**

**1.Project Description:**
   Developed a deep learning model for emotion classification that uses CNNs to analyze facial images and detect happiness. Preprocessed images using OpenCV and 
   Keras's ImageDataGenerator for scaling and augmentation, designed a sequential CNN with multiple convolutional and max-pooling layers, and trained it on a 
   labeled dataset. The model achieves real-time classification by loading new images, pre-processing them, and using the trained model to output predictions.

**2. Key Steps:**
  - **Data Preparation:**
    -  Collected and labeled images of faces.
    -  Rescaled images to 200×200 pixels, normalized pixel values, and applied data augmentation techniques.
 - **Model Architecture:**
   - Built a CNN model with three convolutional layers, each followed by max-pooling layers, using ReLU activations.
   -  Flattened the output and added fully connected layers with a sigmoid activation for binary classification.
 - **Training and Evaluation:**
   - Used a binary cross-entropy loss function and RMSprop optimizer.
   - Achieved an accuracy of around 90% on the validation set.
- **4. Deployment:**
  - Saved the model in Keras format and created a prediction script for real-time classification.
  - Employed a remote model loading function to enable model access from cloud storage.
  **5 . Key Technologies:**
   - Libraries: TensorFlow/Keras, OpenCV, Matplotlib, NumPy
   - Techniques: Data Augmentation, Convolutional Neural Networks, Image Preprocessing
   - Tools: Python, ImageDataGenerator, Model Serialization
- **Accuracy:**
Achieved 90% accuracy on the validation data, demonstrating effective classification of “Happy” and “Not Happy” emotions.
streamlitApp: https://emotion-detection-using-cnn-gbjysbe47ryy68lrr9exat.streamlit.app/

