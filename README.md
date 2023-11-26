# Sentence-Creation
CIS477 Group Project : Sentence Creation

    Overview:
        Our idea was to create a model that would take a given sentence with “masked” word and return a list of words that could potentially replace the masked word. After doing some research we discovered that the best type of model to use for this task would be a BERT model. This model is best for this task because it is designed to "understand the meaning of ambiguous language in text by using surrounding text to establish context". While looking into datasets and models we found a good pre-trained BERT model on hugging face. From there we used this model and another dataset from hugging face to help achieve our end goal. The main issue we faced was figuring out how to load in other dataset to be able to fine-tune the model.

        Source: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model 


    Requirements:
        For this model we did our code and testing in a juptyer notebook and python in a conda environment in visual studio code.
        To be able to run this code yourself you will need to install these python libraries. This can be done in the terminal. In addition to those needs being listed here you can also find a list in the requirements.txt folder of this project.

        We used a pre-trained BERT model from hugging face and to be able to use it we needed to install tokenizer and transformers.
            pip install tokenizer
            pip install transformers

        Even though the model we used was pre-trained we still needed a dataset to use for fine-tuning and evaluating metrics. So we found the 'c4' dataset on hugging face to load and use the dataset we need this library.
            pip install datasets

        Used numpy for the fine-tuning section.
            pip install numpy

        Needed for building, training, and evaluating our machine learning model.
            pip install tensorflow

        In order to present the model and give a front-end view we used streamlit.
            pip install streamlit


    Models:
        For our project we used a pre-trained transformer BERT language model from hugging face. It is pre-trained on two datasets; wikipedia and bookcorpus. The specific model we chose have multiple variations and we chose bert-base-uncased because it was in english and it wasn't case sensitive. (What is it doing)

        Link to the models Hugging Face entry:
        https://huggingface.co/bert-base-uncased 

        Datasets:
            To further evaluate the model and get it to work with our desired task I found another dataset to fine-tune and evaluate metrics with. The dataset we used is "c4" dataset from Hugging Face. This dataset comes in 4 variants and we used the "en" varient. We chose this one because it was in enlish and it had been cleaned; there was a "badwords filter".

            Link to the dataset's Hugging Face entry:
            https://huggingface.co/datasets/c4 

    Metrics: (section left to finish)
        Since the model was pre-trained I just found one additional dataset to be able to fine tune and look at how weel the model did in the training process.


    License:
        There is a license document for the model that can be found in the project under the name 'LICENSE.txt'.

Creators: Avery Harris and Kaylee Odom