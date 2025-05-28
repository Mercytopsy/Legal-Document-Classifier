import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import os


model_folder = "../models"
cat_model = joblib.load(os.path.join(model_folder, "model.pkl"))
label_encoder = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))



# Load tokenizer, model, and label encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)




# Text cleaning and embedding
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("&nbsp;", " ").strip()

def chunk_text(text, chunk_size=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size - 2] for i in range(0, len(tokens), chunk_size - 2)]
    return [['[CLS]'] + chunk + ['[SEP]'] for chunk in chunks]

def embed_chunks(chunks):
    chunk_embeddings = []
    for chunk in chunks:
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        input_tensor = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = bert_model(input_tensor)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            chunk_embeddings.append(cls_embedding.squeeze().cpu().numpy())

    return np.mean(chunk_embeddings, axis=0)

def get_embedding(text):
    chunks = chunk_text(text)
    return embed_chunks(chunks)

def predict_new_cases(df_new):
    df_new['clean_full_report'] = df_new['full_report'].apply(clean_text)
    embeddings = np.array([get_embedding(text) for text in df_new['clean_full_report']])
    y_pred = cat_model.predict(embeddings)
    df_new['predicted_label'] = label_encoder.inverse_transform(y_pred)
    return df_new[['predicted_label']].to_dict(orient='records')