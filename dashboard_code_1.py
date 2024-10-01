import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np


# Carregar dados de ações existentes
df = px.data.stocks()

# Adicionar dados entre 2022-24 de determindas empresas "Big Tech"
additional_dates = pd.date_range(start='2022-09-30', end='2024-09-30', freq='B')
additional_data = {'date': additional_dates, 'GOOG': np.random.uniform(2000, 3000, len(additional_dates)), 
                   'AAPL': np.random.uniform(100, 200, len(additional_dates)),
                   'AMZN': np.random.uniform(3000, 4000, len(additional_dates)), 
                   'TSLA': np.random.uniform(500, 800, len(additional_dates)), 
                   'NVDA': np.random.uniform(500, 700, len(additional_dates))
                   }

df_additional = pd.DataFrame(additional_data)  # Cria um novo dataframe
df = pd.concat([df, df_additional, ], ignore_index=True)  # Concatena os dados tanto existentes como novos para geração do gráfico
app =  dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # Inicialização do app Dash


# Layout do gráfico de linhas dinâmico
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Ações", className="text-center"), className="mb-4 mt-4")]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='stocks-graph'), width=12)]),
    dbc.Row([
        dbc.Col(dbc.Button("Atualizar Gráfico", id='update-button', className="btn btn-primary"), width=12, className="text-center mt-4")
        ])
])


# Callback para interatividade dos elementos visuais
@app.callback(
    Output('stocks-graph', 'figure'), [Input('update-button', 'n_clicks')])

# Função que irá produzir gráficos, controle dos elementos textuais e visuais
def update_graph(n_clicks):
    fig = px.line(df, x='date', y=['GOOG', 'AAPL', 'AMZN', 'TSLA', 'NVDA'], labels={'date': 'Mês/Ano', 'value': 'Preço (USD)', 'variable': 'Empresas'}, 
                  title="Preços das Ações ao Longo do Tempo (2022-2024)")
    fig.update_layout(
        paper_bgcolor='lightgray',  # Cor do fundo externo
        font_color='black')  # Cor da fonte
    return fig


# Rodar o servidor
if __name__ == '__main__':
    app.run_server(debug=True)