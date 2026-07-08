import json
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
JSON_PATH  = os.path.join(BASE_DIR , "settings.json")

def cargar_parametro(clave):
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"No se encontró el archivo: {JSON_PATH}")
        
    with open(JSON_PATH, 'r', encoding='utf-8') as archivo:
        datos = json.load(archivo)
        return datos.get(clave)

# Exportas el parámetro como una variable global de este módulo
RISK_FREE_RATE = cargar_parametro('RISK_FREE_RATE')
ANUAL_PERIOD = cargar_parametro('ANUAL_PERIOD')
ASSETS = cargar_parametro('ASSETS')
WEIGHTS = cargar_parametro('WEIGHTS')
NUM_SIMULATIONS = cargar_parametro('NUM_SIMULATIONS')
