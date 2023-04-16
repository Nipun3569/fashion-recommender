import tensorflow
import numpy as np
from numpy.linalg import norm
import os

from tqdm import tqdm
import pickle


from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()#addition of layer
])

print(model.summary())


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    #converting image to array
    
    
    img_array = image.img_to_array(img)
    
    #by expanding 2d array is converted to 4 d array
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    
      #normalising data
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('dataset'):#joining path of images with file name
    filenames.append(os.path.join('dataset',file))

    #print(len(file_names))
#print(file_names[0:5])
feature_list = []

for file in tqdm(filenames):#each list will have 2048 features for each image
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
