from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
import numpy as np
#import re
#import matplotlib.pyplot as plt
#from datasets import load_dataset
import streamlit as st

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

   # # Real-time Demo
st.header("Sentence Creator")
query = st.text_input("Enter sentence here:")

if query:
    inp = tokenizer(query,return_tensors='tf')

    mask_loc = np.where(inp['input_ids'].numpy()[0] == tokenizer.mask_token_id)[0].tolist()

    st.write(f"Masked Token Position: {mask_loc}")

    out = model.predict([inp['input_ids'], inp['attention_mask']])
    predicted_tokens = np.argmax(out['logits'][0][mask_loc],axis=1)
    
    st.subheader(f"Predicted Words:")

    for word in predicted_tokens:
        predicted_words = tokenizer.decode(word)
        st.write(f'- {predicted_words}')

    # st.subheader(query)
    if(query.__contains__("snow")):
        st.snow()

