
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Frontera Eficiente", layout="wide")
st.title("📈 Frontera Eficiente de Markowitz — App")

# Sidebar inputs
st.sidebar.header("Configuración")
tickers_input = st.sidebar.text_input("Tickers (separados por coma):", "AAPL,MSFT,GOOGL,AMZN")
start_date = st.sidebar.date_input("Fecha inicial", datetime(2022,1,1))
end_date = st.sidebar.date_input("Fecha final", datetime.today())
rf = st.sidebar.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.0) / 100
num_portfolios = st.sidebar.slider("Número de simulaciones", 500, 10000, 5000, step=500)

if st.sidebar.button("Ejecutar análisis"):
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()!='']
    if len(tickers) == 0:
        st.error("Ingrese al menos un ticker válido.")
    else:
        with st.spinner("Descargando datos..."):
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            returns = np.log(data / data.shift(1)).dropna()
            trading_days = 252
            mean_returns = returns.mean() * trading_days
            cov_matrix = returns.cov() * trading_days

        # Functions
        def portfolio_performance(weights, mean_returns, cov_matrix):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, vol

        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
            ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(ret - rf) / vol

        def minimize_volatility(weights, mean_returns, cov_matrix):
            return portfolio_performance(weights, mean_returns, cov_matrix)[1]

        # Optimization
        num_assets = len(tickers)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1.0 / num_assets]

        max_sharpe = minimize(
            neg_sharpe_ratio,
            initial_guess,
            args=(mean_returns, cov_matrix, rf),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        min_var = minimize(
            minimize_volatility,
            initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        ret_sharpe, vol_sharpe = portfolio_performance(max_sharpe.x, mean_returns, cov_matrix)

        # Monte Carlo simulation
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio

        results_df = pd.DataFrame(results.T, columns=['Riesgo', 'Retorno', 'Sharpe'])
        weights_df = pd.DataFrame(weights_record, columns=tickers)
        max_sharpe_idx = results_df['Sharpe'].idxmax()
        max_sharpe_port = results_df.loc[max_sharpe_idx]
        best_weights = weights_df.loc[max_sharpe_idx]

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno esperado", f"{(max_sharpe_port['Retorno']*100):.2f}%")
        col2.metric("Riesgo (volatilidad)", f"{(max_sharpe_port['Riesgo']*100):.2f}%")
        col3.metric("Sharpe Ratio", f"{max_sharpe_port['Sharpe']:.2f}")

        # Plot interactive frontier
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Riesgo'],
            y=results_df['Retorno'],
            mode='markers',
            marker=dict(color=results_df['Sharpe'], colorscale='Viridis', showscale=True, size=6, opacity=0.7),
            name='Portafolios simulados'
        ))
        fig.add_trace(go.Scatter(
            x=[max_sharpe_port['Riesgo']],
            y=[max_sharpe_port['Retorno']],
            mode='markers+text',
            text=['Máx Sharpe'],
            textposition='top center',
            marker=dict(color='red', size=12, symbol='star'),
            name='Máx Sharpe'
        ))
        fig.update_layout(title='Frontera Eficiente', xaxis_title='Riesgo', yaxis_title='Retorno')
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart of optimal weights
        fig_pie = go.Figure(data=[go.Pie(labels=tickers, values=max_sharpe.x)])
        fig_pie.update_layout(title='Composición del Portafolio Óptimo')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Provide CSV download
        results_csv = results_df.copy()
        results_csv['Riesgo (%)'] = results_csv['Riesgo'] * 100
        results_csv['Retorno (%)'] = results_csv['Retorno'] * 100
        csv_bytes = results_csv.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar resultados (CSV)", data=csv_bytes, file_name='frontera_eficiente_results.csv', mime='text/csv')
