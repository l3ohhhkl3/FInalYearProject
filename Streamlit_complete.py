import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

# Model Paths
SAVED_MODEL_FILE = 'zero-dce-saved-model/'
TFLITE_MODEL_FILE = 'zero-dce.tflite'

# Image size dimensions
IMG_HEIGHT = 600
IMG_WIDTH = 400

# Loading the model
ATTACHED_TFLITE_MODEL_FILE = 'zero-dce.tflite'

# Preprocess the image
def preprocess_image(image_path):
    original_image = Image.open(image_path)
    preprocessed_image = original_image.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return original_image, preprocessed_image


# Inference using model
def infer_tflite(image):
    interpreter = tf.lite.Interpreter(model_path=ATTACHED_TFLITE_MODEL_FILE)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    raw_prediction = interpreter.tensor(output_index)
    output_image = raw_prediction()
    
    output_image = output_image.squeeze() * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape((np.shape(output_image)[0], np.shape(output_image)[1], 3))
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image

# Streamlit app
def main():
    st.title('Zero-DCE Image Processing')
    st.write('Upload an image for processing:')
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        original_image, preprocessed_image = preprocess_image(uploaded_file)
        st.image(original_image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Process Image'):
            with st.spinner('Processing...'):
                output_image = infer_tflite(preprocessed_image)
                st.image(output_image, caption='Processed Image', use_column_width=True)
            st.success('Processing completed successfully!')

if __name__ == '__main__':
    main()
