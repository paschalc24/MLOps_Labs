import streamlit as st
import requests

st.title('CIFAR-10 PREDICTION')

img = st.file_uploader("Upload an Image to Classify", type=["png", "jpg"])

if img is not None:
    st.image(img, width=250)

if st.button('Predict'):
    if img is None:
        st.warning("Upload an image")
    else:
        data = {"image": (img.name, img.getvalue(), img.type)} 
        try:
            response = requests.post('https://cifar10-app-25436693860.us-east1.run.app/predict', files=data)
            if response.status_code == 200:
                prediction = response.json()['prediction']
                st.success(f'Predicted class: {prediction}')
            else:
                st.error(f'Error occurred during prediction. Status code: {response.status_code}')
        except requests.exceptions.RequestException as e:
            st.error(f'Error occurred during prediction: {str(e)}')
