# Sentence-Creation
CIS477 Group Project: Sentence Creation


    Overview:
        Our idea was to create a model that would take a given sentence with a “masked” word and return a list of words that could potentially replace the masked word. 
        After doing some research we discovered that the best type of model to use for this task would be a BERT model. 
        This model is best for this task because it is designed to "understand the meaning of ambiguous language in the text by using surrounding text to establish context".
        While looking into datasets and models we found a good pre-trained BERT model on hugging face. 
        From there we used this model and another dataset from hugging face to help achieve our end goal. The main issue we faced was figuring out how to load another 
        dataset to be able to evaluate the metrics of the model.

        Source: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model

    Requirements:
        For this model, we did our code and testing in a juptyer notebook and Python in a conda environment in visual studio code.To be able to run this code yourself you 
        will need to install these python libraries. This can be done in the terminal. In addition to those needs being listed here, you can also find a list in the 
        requirements.txt folder of this project.

        We used a pre-trained BERT model from hugging face and to be able to use it we needed to install tokenizer and transformers.
            pip install tokenizer
            pip install transformers

        Even though the model we used was pre-trained we still needed a dataset to use for evaluating metrics. So we found the 'c4' dataset on hugging face to load and use the dataset we need this library.
            pip install datasets

        Used numpy for metrics section.
            pip install numpy

        Needed for building, training, and evaluating our machine learning model.
            pip install tensorflow

        In order to present the model and give a front-end view we used streamlit.
            pip install streamlit


    Models:
        For our project, we used a pre-trained transformer BERT language model from hugging face.
        It is pre-trained on two datasets; Wikipedia and Bookcorpus. The specific model we chose has multiple variations and we chose bert-base-uncased because it was in English 
        and it wasn't case-sensitive. We then loaded that pre-trained model and ran our real-time deom to see if it could effectively produce options for what could go into the masked area.

        Link to the models Hugging Face entry:
        https://huggingface.co/bert-base-uncased

    Metrics:
        Since the model was pre-trained we looked for an additional dataset to use for reporting evaluation metrics. The dataset we decided to use is the "c4" dataset from Hugging Face.
        This dataset comes in 4 variants and we used the "en" variant. We chose this one because it was in English and it had been cleaned; it had a "badwords filter". For our model, we 
        evaluated the cross entropy loss and accuracy. Both are important measurements to ensure that the model is learning and will be able to predict the mask at a highly effective rate.

        Link to "c4" dataset: https://huggingface.co/datasets/c4

    Sources/Links:
     - https://www.analyticsvidhya.com/blog/2022/09/fine-tuning-bert-with-masked-language-modeling/
     - https://huggingface.co/bert-base-uncased
     - https://huggingface.co/datasets/c4
     - https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model

    License:
        There is a license document for the model that can be found in the project under the name 'LICENSE.txt'.


Creators: Avery Harris and Kaylee Odom

