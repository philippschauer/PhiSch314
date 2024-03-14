# PhiSch314


## Task 1

### Description

In this model, we want to predict the label of a short sequence. We have 5 different labels and around 37.000 samples. However, since they are not distributed equally, we sample the same number of samples from each class.

Next, we embed the words, so they are processable by a neural network, and encode the labels into vectors of length 5. We split into train and test set in order to be able to evaluate our model on unseen data.

Then, we train the model where we try out a few different hyperparameters, evaluate on the test set and save the model.



### Instructions

To run the model, put the file "sample_data_for_task1.csv" into the folder "data" of Task_1. Then run the notebook "2024-03-12 - Train Model.ipynb" after installing all the required packages. After running this notebook, all the required model files, in order to start the API

In a different script, we load the tokenizer, label encoder, and neural network. Then we start the API, i.e. run the script from the directory Task_1

```python api_call.py```

so we can call the model from the following link:

```localhost:8314/docs```

## Task 2

In this task, the goal is to design a concept of a system that reads a PDF document of several products and saves the individual positions in a structured dataframe.

Therefore, I created a document that outlines a mapping of several components working together, first to read the document and extract the text, then an NLP model that is trained on the texts of many similar PDFs.

Finally, we have a script that orders these outputs into a dataframe including the byte-encoded images from the images and stores the final table into our data warehouse.
