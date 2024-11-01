from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf

app = Flask(__name__)

# List of EGX30 stocks
egx30_tickers = [
    'ABUK.CA', 'COMI.CA', 'CIEB.CA', 'ETEL.CA', 'EFG.CA', 'ESRS.CA', 
    'HRHO.CA', 'MNHD.CA', 'SWDY.CA', 'TALAAT.CA', 'AUTO.CA', 'CCAP.CA', 
    'ORAS.CA', 'JUFO.CA', 'ORWE.CA', 'PHDC.CA', 'PACHIN.CA', 'AMER.CA', 
    'MFPC.CA', 'CLHO.CA', 'ISPH.CA', 'SKPC.CA', 'FWRY.CA', 'DCRC.CA', 
    'TAMWEEL.CA', 'ALCN.CA', 'SUGR.CA', 'EGTS.CA', 'BINV.CA', 'EGCH.CA'
]

# Define indicator calculation functions
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = np.maximum(high.diff(), close.shift() - low.diff())
    tr = np.maximum(tr, low.diff())
    tr = tr.rolling(window=window).sum()
    return tr

def compute_momentum(data, window=14):
    return data['Close'].diff(window)

def compute_tsi(data, window=14):
    price_change = data['Close'].diff()
    ema1 = price_change.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return ema2 / ema1

def get_recommendation(data):
    signals = []
    for index in range(len(data)):
        rsi = data['RSI'].iloc[index]
        adx = data['ADX'].iloc[index]
        momentum = data['Momentum'].iloc[index]
        tsi = data['TSI'].iloc[index]
        buy_signals = sum([rsi < 30, adx > 25, momentum > 0, tsi > 0])
        sell_signals = sum([rsi > 70, momentum < 0, tsi < 0])
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            if buy_signals / total_signals >= 0.8:
                signals.append("booming buy")
            elif buy_signals > sell_signals:
                signals.append("buy")
            elif sell_signals > buy_signals:
                signals.append("sell")
            else:
                signals.append("hold")
        else:
            signals.append("hold")
        
        if rsi < 30:
            signals[-1] = "oversold"
        elif rsi > 70:
            signals[-1] = "overbought"

    return signals

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/one_stock', methods=['POST'])
def one_stock():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        recommendation = "No data available for the provided ticker."
    else:
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        recommendation = recommendations[-1]

    return render_template('index.html', recommendation=f"Recommendation for {ticker}: {recommendation}")

@app.route('/summary_indicators', methods=['POST'])
def summary_indicators():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    recommendations_summary = {'oversold': [], 'booming buy': [], 'buy': [], 'overbought': []}

    for ticker in egx30_tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            continue
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        last_recommendation = recommendations[-1]
        if last_recommendation in recommendations_summary:
            recommendations_summary[last_recommendation].append(ticker)

    summary = {key: (value if value else "None") for key, value in recommendations_summary.items()}
    return render_template('index.html', summary=summary)

@app.route('/oversold_to_buy', methods=['POST'])
def oversold_to_buy():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    booming_buy_stocks = []

    for ticker in egx30_tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            continue
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        last_recommendation = recommendations[-1]
        if last_recommendation == "oversold":
            booming_buy_stocks.append(ticker)

    return render_template('index.html', booming_buy_stocks=booming_buy_stocks)

if __name__ == '__main__':
    app.run(debug=True)
