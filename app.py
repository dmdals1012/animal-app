
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import math




def main():
    st.title('반려견의 감정을 알아보자!')
    st.info('반려견의 사진을 업로드 하면, 반려견의 표정을 분석하여 감정을 나타내줍니다.')
    image = st.file_uploader('반려견을 보여주세요!', type=['jpg','png','jpeg','webp'])

    if image is not None:
        st.image(image)
        image = Image.open(image)

        model = load_model("model/keras_model.h5", compile=False)

        class_names = open("model/labels.txt", "r", encoding='UTF-8').readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        st.info(f'반려견은 현재 {math.floor(confidence_score*100*10)/10}% 확률로 {class_name[2:]} 상태입니다.')

if __name__=='__main__':
    main()