# src/data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path to allow imports from 'app'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.config import FEATURE_COLS


def download_data(ticker, start_date, end_date):
    """
    从 Yahoo Finance 下载股票数据
    """
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Transfer the multi-index column names into single index
    data.columns = [col[0] for col in data.columns]
    if data.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}.")
    return data


def create_features(data):
    """
    基于原始 OHLCV 数据创建特征
    """
    print("Creating features...")
    df = data.copy()

    # 1. 价格变化特征
    df['price_change_1d'] = df['Close'].pct_change(1)
    df['price_change_5d'] = df['Close'].pct_change(5)

    # 2. 滚动均值 (SMA)
    df['sma_10d'] = df['Close'].rolling(window=10).mean()
    df['sma_50d'] = df['Close'].rolling(window=50).mean()

    # 3. 相对强弱指数 (RSI)
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))

    # 4. 滚动波动率
    df['volatility_10d'] = df['price_change_1d'].rolling(window=10).std()

    # 5. 成交量变化
    df['volume_change_1d'] = df['Volume'].pct_change(1)

    return df


def create_target(data, future_days=5):
    """
    创建目标变量：未来 N 天的收益率 (回归目标)
    """
    print(f"Creating target variable (future {future_days}d return)...")
    df = data.copy()

    # 使用 shift(-N) 将未来的价格"拉"到当前行
    df[f'future_return_{future_days}d'] = df['Close'].shift(-future_days) / df['Close'] - 1

    return df


def get_processed_data(ticker, start_date, end_date, future_days=5):
    """
    完整的数据处理流程
    """
    # 1. 下载
    raw_data = download_data(ticker, start_date, end_date)

    # 2. 创建特征
    featured_data = create_features(raw_data)

    # 3. 创建目标
    final_data = create_target(featured_data, future_days)

    # 4. 清理 (丢弃因滚动和shift产生的 NaN)
    final_data = final_data.dropna()

    # 5. 确定特征列和目标列
    target_col = f'future_return_{future_days}d'
    feature_cols = FEATURE_COLS  # Use the centralized feature list

    # 确保没有无限值
    final_data = final_data.replace([np.inf, -np.inf], np.nan).dropna()

    print("Data processing complete.")
    return final_data, feature_cols, target_col


if __name__ == "__main__":
    # 这是一个示例，展示如何运行此模块
    ticker = 'AAPL'
    start = '2015-01-01'
    end = '2023-12-31'

    data, features, target = get_processed_data(ticker, start, end)

    print("\nProcessed Data Head:")
    print(data.head())

    print("\nFeature Columns:")
    print(features)

    print("\nTarget Column:")
    print(target)
