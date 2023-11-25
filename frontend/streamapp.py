# Things you need to run
from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import streamlit as st

# Streamlit App (Setup)
st.title("Sentence Creation")

with st.spinner("Loading the Pre-trained Model..."):
    # intializing a BERT model 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

#model.summary()
with st.spinner("Loading the Dataset..."):
    # load in the dataset from Hugging Face
    dataset_name = "c4"
    # choosing which task (english/clean)
    task_name = "en"
    dataset = load_dataset(dataset_name, task_name, split="validation",streaming=True)

    # count = 0
    # for example in dataset:
    #     if count >= 1000:
    #         break
    #     # Process or use the example as needed
    #     print(example["text"])  # Or perform any other operation
    #     count += 1

    #convert dataset to a list so I can fine tune

    dataset_list = list(dataset)

# sampleing subset for demonstration purposes
subset_size = 1000
data_subset = dataset_list[:min(subset_size, len(dataset_list))]

# extracting texts from the subset
texts = [example["text"] for example in data_subset]

with st.spinner("Tokenizing the data"):
    # tokenize the texts
    inputs = tokenizer(texts,max_length=100, padding='max_length', truncation=True, return_tensors="tf")

# Adding Labels (helps with fine-tuning)
inp_ids = []
labels = []
max_label_length = 100
for inp in inputs.input_ids.numpy():
    actual_tokens = list(set(range(100)) - set(np.where((inp == 101) | (inp == 102) | (inp == 0))[0].tolist()))
    num_of_token_to_mask = int(len(actual_tokens) * 0.15)
    token_to_mask = np.random.choice(np.array(actual_tokens), size=num_of_token_to_mask, replace=False).tolist()
    
    label = inp[token_to_mask].copy()
    inp[token_to_mask] = 103

    pad_length = max_label_length - len(label)
    label = np.pad(label, (0, pad_length), mode='constant', constant_values=0)

    inp_ids.append(inp)
    labels.append(label)
    
inp_ids = tf.convert_to_tensor(inp_ids)
labels = tf.convert_to_tensor(labels)
inputs['input_ids'] = inp_ids
inputs['labels'] = labels

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

with st.spinner("Training in Progress..."):
    history = model.fit([inputs.input_ids, inputs.attention_mask], inputs.labels, verbose=1, batch_size=14, epochs=2)

#plotting
losses = history.history['loss']
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(len(losses)),losses)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Epoch vs Loss")
plt.grid()
plt.show()

# Display results on Streamlit
st.subheader("Epoch Vs. Loss")
st.pyplot(fig)

# Final loss number
st.write("Final Loss:", losses[-1])

# Metric Section
st.subheader("Model Metrics")

# with st.spinner("Evaluating on training data..."):
#     train_metrics = model.evaluate([inputs.input_ids, inputs.attention_mask], inputs.labels)
# st.write("Training Data Metrics:")
# st.write("Loss:", train_metrics[0])

# Real-time Demo
st.subheader("Real-Time Demonstration")

#input section
user_query = st.text_input("Enter your query:")
if user_query:
    # tokenize the user input
    inp = tokenizer(user_query, return_tensors='tf')
    mask_loc = np.where(inp.input_ids.numpy()[0] == 103)[0].tolist()

    # model prediction
    out = model(inp).logits[0].numpy()
    predicted_tokens = np.argmax(out[mask_loc], axis=1).tolist()

    # decode and display predicted tokens
    predicted_text = tokenizer.decode(predicted_tokens)
    st.write("Predicted Text", predicted_text)

