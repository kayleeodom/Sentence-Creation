# Sentence-Creation
CIS477 Group Project (Avery Harris and Kaylee Odom)


    Problem/Task:
        Create or work with a machine learning model that takes a given sentence with a "masked" word and returns a list of words that could potentially replace the masked word.


    Summary

    Requirements
        Some things you need to have to be able to run the project. A more broken down list can be found in the requirements folder or this project.

        pip install tokenizer

        pip install transformers

        pip install datasets

        pip install numpy

        pip install tensorflow

        In order to present the model and give a front-end view we used streamlit. Streamlit is a ....
        pip install streamlit


    Models
        For our project we used a pre-trained BERT language model from hugging face. It is pre-trained on two datasets; wikipedia and bookcorpus.

    Metrics
        Since the model was pre-trained I just found one additional dataset to be able to fine tune and look at how weel the model did in the training process.
        how well the model did on the training data and eval data; used a seperate dataset from hugging face for fine-tuning and seeing how well the model did.

