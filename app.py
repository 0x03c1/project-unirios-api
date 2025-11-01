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


def tendencia(slope: float) -> str:
    if slope > 0.02:
        return "em alta"
    elif slope < -0.02:
        return "em queda"
    else:
        return "estável/oscilando"


def computer_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            'last_data': pd.NaT,
            'last_value': float('nan'),
            'mean12': float('nan'),
            'ups12': 0,
            'std12': float('nan'),
            'slope24': float('nan'),
            'trend24': 'indisponível',
            'max36': float('nan'),
            'min36': float('nan'),
            'dmin36': pd.NaT,
            'dmax36': pd.NaT,
            'position': 'indisponível'
        }

    last_data = df['data'].iloc[-1]
    last_value = df['valor'].iloc[-1]

    ult12 = df.tail(12).copy()
    mean12 = float(ult12['valor'].mean())
    std12 = float(ult12['valor'].std())
    ups12 = int((ult12['valor'].diff() > 0).sum())

    ult24 = df.tail(24).copy()
    if len(ult24) >= 3:
        x = np.arange(len(ult24))
        slope = float(np.polyfit(x, ult24['valor']
                                 .values, 1)[0])
    else:
        slope = 0.0
    trend = tendencia(slope)

    ult36 = df.tail(36).copy()
    max36 = float(ult36['valor'].max())
    min36 = float(ult36['valor'].min())
    dmax36 = ult36.loc[ult36['valor'].idxmax(), 'data']
    dmin36 = ult36.loc[ult36['valor'].idxmin(), 'data']

    if std12 > 1e-9 and last_value > mean12 + 0.5 * std12:
        pos = "acima da média"
    elif std12 > 1e-9 and last_value < mean12 - 0.5 * std12:
        pos = "abaixo da média"
    else:
        pos = "na média"

    return {
        'last_data': last_data,
        'last_value': last_value,
        'mean12': mean12,
        'ups12': ups12,
        'std12': std12,
        'slope24': slope,
        'trend24': trend,
        'max36': max36,
        'min36': min36,
        'dmin36': dmin36,
        'dmax36': dmax36,
        'position': pos
    }


def create_figure(df: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title=f"{titulo} - Dados indisponíveis",
            xaxis_title="Data",
            yaxis_title="Valor",
        )
        return fig

    df_plot = df.copy()
    df_plot['mm12'] = df_plot['valor'].rolling(12, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=df_plot['data'],
        y=df_plot['valor'],
        mode='lines+markers',
        name='Valor Mensal',
        )
    )

    fig.add_trace(go.Scatter(
        x=df_plot['data'],
        y=df_plot['mm12'],
        mode='lines+markers',
        name='Média Mensal 12 meses',
        )
    )

    last12 = df_plot.tail(12)
    fig.add_trace(
        go.Bar(
            x=last12['data'],
            y=last12['valor'],
            name='Últimos 12 meses',
            opacity=0.5
        )
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="mês/ano",
        yaxis_title="valor",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


app = Dash(__name__)
app.layout = html.Div([
    html.H1("Análise de Séries Temporais do Banco Central do Brasil"),
    dcc.Input(
        id='serie-input',
        type='number',
        value=1,
        min=1,
        step=1,
        placeholder="Digite o número da série SGS"
    ),
    html.Button('Buscar', id='fetch-button', n_clicks=0),
    html.Div(id='stats-output'),
    dcc.Graph(id='time-series-graph')
])


if __name__ == '__main__':
    app.run(debug=True)
