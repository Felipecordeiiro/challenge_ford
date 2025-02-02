from typing import Optional
from modules.get_data import getting_data
from modules.clean_data import clean_data, dowloand_cleaned_data
from modules.process_data import process_data, dowloand_processed_data

def main(y_inicio:int, y_fim:Optional[int]):
    all_data = getting_data(y_inicio, y_fim)
    all_data_cleaned = clean_data(all_data)
    all_data_processed = process_data(all_data_cleaned)

    for i, year in enumerate(range(y_inicio, y_fim+1)):
        dowloand_cleaned_data(all_data_cleaned[i], year)
        dowloand_processed_data(all_data_processed[i], year)
        
main(2014,2020)