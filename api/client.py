import requests
import streamlit as st

def get_llama_response(text):
    response = requests.post(
        'http://localhost:8000/topic/invoke',
        json={'input': {'topic': text }}
    )

    return response.json()['output']['content']

st.title('Test FastApi Llama')
input_text = st.text_input("Write some topic")

if input_text:
    st.write(get_llama_response(input_text))