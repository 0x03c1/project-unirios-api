import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


def build_api_url(serie_sgs: int):
    base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{int(serie_sgs)}/dados"
    today = datetime.today().strftime("%d/%m/%Y")
    return f"{base}?formato=json&dataInicial=01/01/2013&dataFinal={today}"


def fetch_data(serie_sgs: int):
    url = build_api_url(serie_sgs)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = json.loads(response.text or '[]')
    df = pd.DataFrame(data)
    if df.empty or 'data' not in df or 'valor' not in df:
        return pd.DataFrame(columns=['data', 'valor'])
    
    df['data'] = pd.to_datetime(df['data'],
                                format="%d/%m/%Y",
                                errors='coerce')
    df['valor'] = (
        df['valor'].astype(str)
        .str.replace('.', '', regex=False).astype(float)
    )
    df = df.dropna(subset=['data']).sort_values('data').reset_index(drop=True)
    return df


def tendencia(slope: float):
    if slope > 0.02:
        return "em alta"
    elif slope < -0.02:
        return "em queda"
    else:
        return "estável/oscilando"
