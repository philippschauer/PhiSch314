from fastapi import FastAPI
import numpy as np
from gensim.models import Word2Vec
import pickle
import keras
import uvicorn

# Load all the models and tokenizers
with open('utils/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('utils/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

model = keras.models.load_model('utils/model.h5')

word2vec_model = Word2Vec.load('utils/word2vec_model.bin')

# Initialize FastAPI app
app = FastAPI()

# Define the function that embeds the input
def text_to_embeddings(text, word2vec_model):
    tokens = tokenizer(text)
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0) 
    else:
        return np.zeros(50)


# Predict the label of the word  
@app.get('/predict')
def predict_word(word):
    new_word_embedding = text_to_embeddings(word, word2vec_model).reshape(1, -1) 

    # Predict the label for the embedded word
    predicted_probabilities = model.predict(new_word_embedding)
    predicted_label_index = predicted_probabilities.argmax(axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])
    return {'predicted_label': predicted_label[0]}


# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8314)