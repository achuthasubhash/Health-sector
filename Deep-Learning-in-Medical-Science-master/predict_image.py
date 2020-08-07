## Predicitng Models


from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_vgg19.h5') #load model
img = image.load_img('val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(224, 224)) #load image
x = image.img_to_array(img) #to array
x = np.expand_dims(x, axis=0) #expand to 4D bec predict require 4d
img_data = preprocess_input(x) #preprocess i/p
classes = model.predict(img_data) #predicting
