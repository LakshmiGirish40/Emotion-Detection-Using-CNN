
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
#image data generator is the package to lable the images & it will automatically lable all the images
img = image.load_img(r"D:\pictures\testing\IMG_1691.JPG")
plt.imshow(img)
i1 = cv2.imread(r"D:\pictures\testing\IMG_1691.JPG")
i1
# 3 dimension metrics are created for the image
# Display the array to confirm it's loaded
print(i1)

# Check the shape to confirm the 3D structure (height, width, channels)
if i1 is not None:
    print("Image shape:", i1.shape)
else:
    print("Image could not be loaded. Please check the file path.")
    
    
train = ImageDataGenerator(rescale = 1/200)
validataion = ImageDataGenerator(rescale = 1/200)
# to scale all the images i need to divide with 255
# we need to resize the image using 200, 200 pixel

    
train_dataset = train.flow_from_directory(r"D:\pictures\training",
                                         target_size = (200,200),
                                         batch_size = 20,
                                         class_mode = 'binary')
validataion_dataset = validataion.flow_from_directory(r'D:\pictures\validation',
                                          target_size = (200,200),
                                          batch_size = 20,
                                          class_mode = 'binary')

train_dataset.class_indices
train_dataset.classes

# now we are applying maxpooling

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001),
              metrics = ['accuracy']
              )


dir_path = r"D:\pictures\testing"
for i in os.listdir(dir_path ):
    print(i)
    #img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
   # plt.imshow(img)
   # plt.show()
dir_path = r"D:\pictures\testing"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
# Directory path containing the images
dir_path = r"D:\pictures\testing"
for i in os.listdir(dir_path):
    # Construct the image path and load the image
    img_path = os.path.join(dir_path, i)
    img = image.load_img(img_path, target_size=(200, 200))
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Optional: hide axes
    plt.show()

    x= image.img_to_array(img) / 255.0
    x=np.expand_dims(x,axis = 0)
    images = np.hstack([x])

    val = model.predict(images)
    
    if val<0.5:
        print( 'i am happy')
    else:
        print('i am not happy')