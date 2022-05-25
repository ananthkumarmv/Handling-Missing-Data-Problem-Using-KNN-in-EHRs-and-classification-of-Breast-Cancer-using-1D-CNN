# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:08:52 2022

@author: Unbeknownstguy
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pandas as pd

# st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.set_page_config(page_title="Breast Cancer Prediction")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# state = _get_state()

# state.page_config = st.set_page_config(
#     page_title="BPJV SI Database Manager test",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

components.html(
    """
    <!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {box-sizing: border-box;}
nav {
overflow: hidden;
background-color: #330b7c;
padding: 10px;
}
.links {
font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
font-weight: bold;
float: left;
color:white;
text-align: center;
padding: 12px;
text-decoration: none;
font-size: 18px;
line-height: 25px;
border-radius: 4px;
}
nav .logo {
font-size: 25px;
font-weight: bold;
}
nav .links:hover {
background-color: rgb(214, 238, 77);
color: rgb(42, 10, 94);
}
nav .selected {
background-color: dodgerblue;
color: white;
}
.rightSection {
float: right;
}
@media screen and (max-width: 870px) {
nav .links {
float: none;
display: block;
text-align: left;
}
.rightSection {
float: none;
}
}
</style>
</head>
<body>
<nav>
<a class="links logo" href="Image/cancer.png">Breast Cancer Tumor Prediction</a>
<div class="rightSection">
<a class="selected links" href="h">Home</a>
<a class="links" href="#">Contact Us</a>
<a class="links" href="#">About Us</a>
<a class="links" href="#">More Info</a>
<a class="links" href="#">Donate</a>
</div>
</nav>
</body>
<!-- Footer -->
<footer class="page-footer font-small blue">

  <!-- Copyright -->
  <div class="footer-copyright text-center py-3">© 2020 Copyright:
    <a href="/"> MDBootstrap.com</a>
  </div>
  <!-- Copyright -->

</footer>
<!-- Footer -->

</html>
    """,
    height=80,
)






# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

# local_css("C:/Users/Unbeknownstguy/Documents/GitHub/Projects/Machine_Learning/Handling-Missing-Data-Problem-Using-KNN-in-EHRs-for-Cancer-Prediction/3_CNN/style.css")
# remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# # icon("search")
# selected = st.text_input("", "Search...")
# button_clicked = st.button("OK")


# c1, c2, c3, c4, c5 = st.columns(5)

# c1.button("Home")
# c2.button("About")
# c3.button("News")
# c4.button("Donate")
# c5.button("Contact")




#loading saved model
loaded_model = keras.models.load_model('C:/Users/Unbeknownstguy/Documents/GitHub/Projects/Machine_Learning/Handling-Missing-Data-Problem-Using-KNN-in-EHRs-for-Cancer-Prediction/3_CNN/model.h5')

def Prediction(input_data):
    
    cancer_dataset = pd.read_csv('C:/Users/Unbeknownstguy/Documents/GitHub/Projects/Machine_Learning/Handling-Missing-Data-Problem-Using-KNN-in-EHRs-for-Cancer-Prediction/Dataset/data.csv')
    
    columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'compactness_worst', 'concavity_worst']
    
    X_new = pd.DataFrame(cancer_dataset, columns=columns)
    
    scaler = StandardScaler()
    
    scaler.fit_transform(X_new)
    
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshape)

    prediction = loaded_model.predict(std_data)

    if prediction[0]<=0.5:
        return "Benign"
    else:
        return "Malignant"
        
        
        
def main():
    
    
    col1, col2 = st.columns(2)

    radius_mean = col1.text_input('Radius Mean')
    try:
        if 6.9810 <= float(radius_mean) <= 28.1100:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    texture_mean = col1.text_input('Texture Mean')
    try:
        if 9.7100 <= float(texture_mean) <= 39.2800:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    perimeter_mean = col1.text_input('Perimeter Mean')
    try:
        if 43.7900 <= float(perimeter_mean) <= 188.5000:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    area_mean = col1.text_input('Area Mean')
    try:
        if 143.5000 <= float(area_mean) <= 2501.0000:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    concavity_mean = col1.text_input('Concavity Mean')
    try:
        if 0.0000 <= float(concavity_mean) <= 0.4268:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    radius_se = col1.text_input('Radius Se')
    try:
        if 0.1115 <= float(radius_se) <= 2.8730:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    perimeter_se = col1.text_input('Perimeter Se')
    try:
        if 0.7570 <= float(perimeter_se) <= 21.9800:
            pass
        else:
            col1.error("Not in range")
    except ValueError:
        pass


    area_se = col2.text_input('Area Se')
    try:
        if 6.8020 <= float(area_se) <= 542.2000:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    radius_worst = col2.text_input('Radius Worst')
    try:
        if 7.9300 <= float(radius_worst) <= 36.0400:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    texture_worst = col2.text_input('Texture Worts')
    try:
        if 12.0200 <= float(texture_worst) <= 49.5400:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    perimeter_worst = col2.text_input('Perimeter worst')
    try:
        if 50.4100 <= float(perimeter_worst) <= 251.2000:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    area_worst = col2.text_input('Area Wrost')
    try:
        if 185.2000 <= float(area_worst) <= 4254.0000:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    compactness_worst = col2.text_input('Compactness Wrost')
    try:
        if 0.0273 <= float(compactness_worst) <= 1.0580:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass


    concavity_worst = col2.text_input('Concavity Worst')
    try:
        if 0.0000 <= float(concavity_worst) <= 1.2520:
            pass
        else:
            col2.error("Not in range")
    except ValueError:
        pass
    
    
    # code for prediction
    diagnosis = ''
    
    try:
        if st.button('Result'):
            diagnosis = Prediction([radius_mean, texture_mean, perimeter_mean, area_mean, concavity_mean, radius_se, perimeter_se, area_se, radius_worst, texture_worst, perimeter_worst, area_worst, compactness_worst, concavity_worst])
                    
        st.success(diagnosis)
    
    except ValueError:
        st.error("Cells Should not be empty")
    
    
if __name__ == "__main__":
    main()

