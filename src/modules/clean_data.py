from dotenv import load_dotenv
from scipy import stats
import polars as pl
import numpy as np
import re
import os

# PATH DATA
CLEANED_DATA_PATH = os.getenv('CLEANED_DATA')

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
        pl.col('dateOfIncident').str.strptime(pl.Datetime, fmt='%Y-%m-%d', strict=False).alias('dateOfIncident'),
        pl.col('dateComplaintFiled').str.strptime(pl.Datetime, fmt='%Y-%m-%d', strict=False).alias('dateComplaintFiled')
    ])
    return data


def remove_stop_words(data:pl.DataFrame) -> pl.DataFrame:
    punctuation_pattern = r'[^\w\s]'
    text_cleaned = re.sub(punctuation_pattern, '', data)
    
    return data

def remove_duplicates(data:pl.DataFrame) -> pl.DataFrame:
    data = data.drop_duplicates()
    return data

def remove_blank_spaces_and_convert_to_lower(data:pl.DataFrame) -> pl.DataFrame:
    text_columns = ['manufacturer', 'components', 'summary', 'type', 'productMake', 'productModel']
    data[text_columns] = data[text_columns].apply(lambda x: x.str.lower().str.strip())
    return data

def remove_stop_words(data:pl.DataFrame) -> pl.DataFrame:
    # odiNumber,manufacturer,crash,fire,numberOfInjuries,numberOfDeaths,dateOfIncident,dateComplaintFiled,vin,components,summary,type,productYear,productMake,productModel
    print(data)

def clean_data(all_data:list[pl.DataFrame]) -> list[pl.DataFrame]:
    data_cleaned = []
    for data in all_data:
        data = data_to_datetime(data)
        data = remove_duplicates(data)
        data = remove_blank_spaces_and_convert_to_lower(data)
        data = remove_stop_words(data)

        data_cleaned.append(data)
    
    return data_cleaned
