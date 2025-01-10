import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
model= load_model('model.keras')
emotion_labels=['Angry','Disgust','Fear','Happy','sad']
st.title('Emotion Classification App')
st.write('Upload an image to classify the amotions')
Uploaded_file=st.file_uploader('choose an emage....',type=['jpg','jpeg','png'])
if Uploaded_file is not None:
   img=Image .open(Uploaded_file)
   st.image(img,caption='Uploaded Image',use_column_width=True)
   img=img.convert("L")
   img=img.resize((48,48))
   img_array=np.array(img)/255.0
   img_array=np.expand_dims(img_array,axis=-1)
   img_array=np.expand_dims(img_array,axis=0)
   prediction=model.predict(img_array)
   predicted_emotion=emotion_labels[np.argmax(prediction)]
   st.write(f'Predicted Emotion:{predicted_emotion}')







