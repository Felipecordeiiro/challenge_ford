from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import polars as pl
import os

load_dotenv()

# Carregar o tokenizer e modelo BERT
API_TOKEN_HUGGING_FACE=os.getenv('API_TOKEN_HUGGING_FACE')
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, token=API_TOKEN_HUGGING_FACE)
bert_model = BertModel.from_pretrained(MODEL_NAME, token=API_TOKEN_HUGGING_FACE)

# PATH DATA
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA')

# Função para converter texto em embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def generate_embeddings(data:pl.DataFrame) -> pl.DataFrame:
    # Aplicar embeddings no dataset (convertendo para lista antes)
    embeddings = [get_bert_embedding(text) for text in data["summary"].to_list()]
    
    # Transformar rótulos em valores numéricos
    severity_mapping = {"leve": 0, "moderado": 1, "grave": 2}
    
    data = data.with_columns([
        pl.Series("embeddings", embeddings),  # Adiciona os embeddings
        pl.col("severity").map_dict(severity_mapping).alias("severity_label")  # Mapeia os rótulos de severidade
    ])

    return data

def dowloand_processed_data(response:dict, year:int) -> pl.DataFrame:    
    data_response = pl.DataFrame(response)
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    data_response.write_csv(os.path.join(PROCESSED_DATA_PATH, str(year)+".csv"))
    print(f"Dataset procesado salvo em {PROCESSED_DATA_PATH}")

    return data_response

# Função para converter texto em embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def classify_severity_feature(injuries, deaths):
    if deaths > 0:
        return "grave"
    elif injuries > 2:
        return "moderado"
    else:
        return "leve"

def create_severity_feature(data:pl.DataFrame) -> pl.DataFrame:
    """
    Função responsável por criar uma nova feature que armazena informações de severidade, uma target, para um problema de regressão.
    """
    severity_values = [
        classify_severity_feature(injuries, deaths)
        for injuries, deaths in zip(data["numberOfInjuries"].to_list(), data["numberOfDeaths"].to_list())
    ]
    
    data = data.with_columns(
        pl.Series("severity", severity_values)
    )
    
    return data

def encode_features(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns([
        pl.col('crash').cast(pl.Int32),
        pl.col('fire').cast(pl.Int32)
    ])
    
    # Codificar a coluna 'components' usando LabelEncoder
    label_encoder = LabelEncoder()
    encoded_components = label_encoder.fit_transform(data['components'].to_list())

    # Adicionar a nova coluna ao DataFrame Polars
    data = data.with_columns(
        pl.Series('components_encoded', encoded_components)
    )

    return data

def process_data(all_data:list[pl.DataFrame]) -> list[pl.DataFrame]:
    data_cleaned = []
    for data in all_data:
        data = encode_features(data)
        data = create_severity_feature(data)
        data_cleaned.append(data)
    
    return data_cleaned