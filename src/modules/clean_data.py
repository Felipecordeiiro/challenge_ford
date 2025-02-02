from dotenv import load_dotenv
from nltk.corpus import  stopwords
import polars as pl
import os

load_dotenv()

# PATH DATA
CLEANED_DATA_PATH=os.getenv('CLEANED_DATA')

def dowloand_cleaned_data(response:dict, year:int) -> pl.DataFrame:
    
    data_response = pl.DataFrame(response)
    
    if not os.path.exists(CLEANED_DATA_PATH):
        os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

    data_response.write_csv(os.path.join(CLEANED_DATA_PATH, str(year)+".csv"))
    print(f"Dataset limpo salvo em {CLEANED_DATA_PATH}")

    return data_response

def data_to_datetime(data:pl.DataFrame) -> pl.DataFrame:
    # Converter colunas de data para datetime
    data = data.with_columns([
        pl.col('dateOfIncident').str.strptime(pl.Datetime, format='%Y-%m-%d', strict=False).alias('dateOfIncident'),
        pl.col('dateComplaintFiled').str.strptime(pl.Datetime, format='%Y-%m-%d', strict=False).alias('dateComplaintFiled')
    ])
    return data

def remove_ponctuation_special_chars(data:pl.DataFrame) -> pl.DataFrame:
    punctuation_pattern = r'[^\w\s]'
    data = data.with_columns(
        pl.col("summary").str.replace_all(punctuation_pattern, "").alias("summary")
    )
    return data

def remove_blank_spaces_and_convert_to_lower(data:pl.DataFrame) -> pl.DataFrame:
    text_columns = ['manufacturer', 'components', 'summary', 'type', 'productMake', 'productModel']
    # Aplicar lower() e strip() nas colunas de texto
    data = data.with_columns([
        pl.col(col).str.strip_chars().str.to_lowercase().alias(col) for col in text_columns
    ])
    return data

def clean_data(all_data:list[pl.DataFrame]) -> list[pl.DataFrame]:
    data_cleaned = []
    for data in all_data:
        data = data_to_datetime(data)
        data = remove_blank_spaces_and_convert_to_lower(data)
        data = remove_ponctuation_special_chars(data)

        data_cleaned.append(data)
    
    return data_cleaned
