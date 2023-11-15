import pandas as pd
from code import api_request, data_processing
import sys

def automatic_loading(url: str):
    """
    La función se encarga de obtener la información desde una url
    para posteriormente procesarla y guardarla en un csv, listo
    para el analisis estadistico de los datos.

    param: url(str)
    return: none
    """

    # Obtenemos los datos con la url y almacenamos en un csv base
    api_request(url)

    # Convertimos el csv en un DataFrame
    brute_data_df = pd.read_csv('data.csv')

    # Categorizamos los grupos y exportamos como csv con la siguiente función
    data_processing(brute_data_df)
    print('The process of load has been completed successfully')


url = sys.argv[1]
automatic_loading(url)
