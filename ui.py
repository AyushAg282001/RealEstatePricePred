import streamlit as st
import pickle
import pandas as pd
import numpy as np

df = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('Bengaluru_House_Data_Model.pkl', 'rb'))
df12 = pickle.load(open('data12.pkl', 'rb'))

st.title('House Price Prediction')

# Set the background color using CSS
background_color = "#fdf2e9"  # Light gray

background_style = f"""
<style>
    body {{
        background-color: {background_color};
    }}
</style>
"""

# Apply the background style using Markdown
st.markdown(background_style, unsafe_allow_html=True)

# Get unique locations
loca = df['location'].unique()

# Location selectbox
location = st.selectbox('Select Location', loca)

# Number of bathrooms selectbox
l = [1, 2, 3, 4, 5, 6, 7, 8]
bath = st.selectbox('Number of Bathrooms', l)

# Area of the plot input
total_sqft = st.number_input('Area of the Plot',min_value=200, step=1)

# BHK selectbox
BHK = st.selectbox('BHK', l)

x = df12.drop('price', axis='columns')

def predictPrice(location, sqft, bath, bhk):
    loc_idx = np.where(x.columns == location)[0][0]
    temp = np.zeros(len(x.columns))
    temp[0] = sqft
    temp[1] = bath
    temp[2] = bhk
    
    if loc_idx >= 0:
        temp[loc_idx] = 1
    
    return model.predict([temp])[0]

if st.button('Show Price'):
    result = predictPrice(location, total_sqft, bath, BHK)
    st.header("Predicted Price = " + str(int(result)) + " Lakhs")
