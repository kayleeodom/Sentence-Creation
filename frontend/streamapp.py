# Things you need to run
from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import streamlit as st

# Streamlit App (Setup)
st.title("Sentence Creation")

@st.cache_data
def load_model_and_tokenizer():
    # intializing a BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

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

# Compile and fine-tune the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

with st.spinner("Training in Progress..."):
    history = model.fit([inputs.input_ids, inputs.attention_mask], inputs.labels, verbose=1, batch_size=14, epochs=4)

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

with st.spinner("Evaluating on training data..."):
    train_metrics = model.evaluate([inputs.input_ids, inputs.attention_mask], inputs.labels)
st.write("Training Data Metrics:")
st.write("Loss:", train_metrics)

# # Real-time Demo
# st.subheader("Real-Time Demonstration")

# #input section
# user_query = st.text_input("Enter your sentencce with a [MASK] token:")
# if user_query:
#     # tokenize the user input
#     inp = tokenizer(user_query, return_tensors='tf')
#     mask_indices = tf.where(inp['input_ids'][0] == tokenizer.mask_token_id).numpy()

#     if len(mask_indices) > 0:
#         st.write(f"Detected {len(mask_indices)} [MASK] token(s) in the input.")
#         st.write("Predicted words for each [MASK] token:")

#         for idx, mask_index in enumerate(mask_indices, 1):
#             # Model prediction
#             out = model(inp).logits[0].numpy()
#             predicted_tokens = np.argsort(out[mask_index[0]])[::-1][:5]  # Get top 5 predicted tokens

#             # Decode and display predicted tokens
#             predicted_words = [tokenizer.decode(token) for token in predicted_tokens]
#             st.write(f"{idx}. {predicted_words}")

#     else:
#         st.write("No [MASK] token detected in the input. Please include a [MASK] token for prediction.")

