import streamlit as st
import requests

st.title("Upload Lung X-Ray")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post("http://localhost:5000/upload", files=files)

    if response.status_code == 200:
        st.success("File uploaded successfully")
    
        file_name = uploaded_file.name
        predict_response = requests.get(f"http://localhost:5000/predict_cnn/{file_name}")

        if predict_response.status_code == 200:
            result = predict_response.json()
            st.write("<div style='color: red;font-size: 30px;'>Predicted Class: "+result['predicted_class']+"</div>", unsafe_allow_html=True)
        else:
            st.error(f"Error in prediction: {predict_response.json().get('error', 'Unknown error')}")

    else:
        st.error(f"Error: {response.json()['error']}")