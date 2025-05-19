import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Previsão de Ações com Prophet", layout="wide")
st.title("Previsão de Ações com Prophet")

st.markdown("""
Esta aplicação permite fazer previsões de preços de ações utilizando o modelo **Prophet**.

1. Faça o upload de um arquivo **CSV ou Excel (.xlsx)** contendo as colunas `ticker`, `ref.date`, `price.close`.
2. Selecione o ticker desejado.
3. Veja a previsão para os próximos 30 dias.
""")

uploaded_file = st.file_uploader("Escolha o arquivo CSV ou XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df_full = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            df_full = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado.")
            st.stop()

        if not {'ticker', 'ref.date', 'price.close'}.issubset(df_full.columns):
            st.error("O arquivo deve conter as colunas: TICKER, DATE e CLOSE.")
        else:
            tickers = df_full['ticker'].unique().tolist()
            selected_ticker = st.selectbox("Selecione o ticker:", sorted(tickers))

            df = df_full[df_full['ticker'] == selected_ticker][['ref.date', 'price.close']].copy()
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])

            st.subheader(f"Série Histórica - {selected_ticker}")
            st.line_chart(df.set_index('ds')['y'])

            if st.button("Gerar previsão com Prophet"):
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                st.subheader("Previsão para os próximos 30 dias")
                fig = plot_plotly(model, forecast)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")