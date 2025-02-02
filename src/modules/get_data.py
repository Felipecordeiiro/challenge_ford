from typing import Optional
from dotenv import load_dotenv
import polars as pl
import requests
import os

load_dotenv()
api_url = os.getenv('API_URL')

# PATH DATA
RAW_DATA_PATH = os.getenv('RAW_DATA')

def dowloand_raw_data(response:dict|list[dict], year:int) -> pl.DataFrame:
    
    data_response = pl.DataFrame(response)
    
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    products_data = []
    
    for line in data_response["products"]:
        products_data.append(line[0])

    data_response = data_response.drop("products")
    pl_produtcs = pl.DataFrame(products_data)
    pl_produtcs = pl_produtcs.drop("manufacturer")
    data_merged = pl.concat([data_response, pl_produtcs], how="horizontal")
    data_merged.write_csv(os.path.join(RAW_DATA_PATH, str(year)+".csv"))
    print(f"Dataset original salvo em {RAW_DATA_PATH}")

    return data_response

def getting_data(y_inicio:int, y_fim:Optional[int]) -> pl.DataFrame:
    if not y_fim:
        params = {
            "modelYear":y_inicio,
            "make": "acura",
            "model": "rdx"
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            print("Os dados foram aquisicionados com sucesso!")
            response_results = response["results"]
            data = dowloand_raw_data(response_results, y_inicio)
            return data
        
    else:
        all_data = []
        for data in range(y_inicio, y_fim+1):
            params = {
                "modelYear":data,
                "make": "acura",
                "model": "rdx"
            }
            response = requests.get(api_url, params=params)

            if response.status_code == 200:
                print("Os dados foram aquisicionados com sucesso!")
                response = response.json()
                response_results = response["results"]
                data = dowloand_raw_data(response_results, data)
                all_data.append(data)
                
        return data