import random
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from numpy import unique
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    from yahooquery import Ticker
except Exception:
    Ticker = None

from config import (
    AVAILABLE_LLM_MODES,
    CANDIDATE_POOL_K,
    LLM_MODEL_NAME,
    SENTENCE_EMBEDDING_MODEL,
    TOP_K,
)
from data_loader import load_qa
from embedding_store import init_embedding_store
from llm_model import load_llm
from rag import evaluate_stock_lynch_agent, get_rag_answer

warnings.filterwarnings("ignore", category=Warning)
random.seed(42)

st.set_page_config(
    page_title="Peter Lynch Agentic RAG Investment Advisor",
    page_icon="📈",
    layout="wide",
)

# --- Default Dow 30 Constituents & Complete Fundamental Snapshot ---
DEFAULT_DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

FALLBACK_FINANCIALS = {
    "AAPL": {"currentPrice": 228.5, "targetMeanPrice": 252.0, "targetHighPrice": 280.0, "targetLowPrice": 200.0, "recommendationMean": 1.8, "recommendationKey": "buy", "numberOfAnalystOpinions": 42, "totalCash": 65170000000, "totalCashPerShare": 4.31, "ebitda": 134660000000, "totalDebt": 106620000000, "quickRatio": 0.83, "currentRatio": 0.95, "totalRevenue": 391035000000, "debtToEquity": 151.8, "revenuePerShare": 25.48, "returnOnAssets": 0.225, "returnOnEquity": 1.38, "grossProfits": 180683000000, "freeCashflow": 108807000000, "operatingCashflow": 118254000000, "earningsGrowth": 0.112, "revenueGrowth": 0.061, "grossMargins": 0.462, "ebitdaMargins": 0.344, "operatingMargins": 0.315, "profitMargins": 0.240, "financialCurrency": "USD"},
    "MSFT": {"currentPrice": 425.0, "targetMeanPrice": 495.0, "targetHighPrice": 550.0, "targetLowPrice": 440.0, "recommendationMean": 1.4, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 50, "totalCash": 78430000000, "totalCashPerShare": 10.55, "ebitda": 136550000000, "totalDebt": 67120000000, "quickRatio": 1.15, "currentRatio": 1.27, "totalRevenue": 254190000000, "debtToEquity": 23.4, "revenuePerShare": 34.20, "returnOnAssets": 0.148, "returnOnEquity": 0.356, "grossProfits": 176200000000, "freeCashflow": 74070000000, "operatingCashflow": 118548000000, "earningsGrowth": 0.145, "revenueGrowth": 0.160, "grossMargins": 0.694, "ebitdaMargins": 0.537, "operatingMargins": 0.446, "profitMargins": 0.356, "financialCurrency": "USD"},
    "NVDA": {"currentPrice": 138.6, "targetMeanPrice": 168.0, "targetHighPrice": 200.0, "targetLowPrice": 130.0, "recommendationMean": 1.3, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 56, "totalCash": 38480000000, "totalCashPerShare": 1.57, "ebitda": 72510000000, "totalDebt": 10220000000, "quickRatio": 3.50, "currentRatio": 4.17, "totalRevenue": 113270000000, "debtToEquity": 17.2, "revenuePerShare": 4.61, "returnOnAssets": 0.558, "returnOnEquity": 1.152, "grossProfits": 85910000000, "freeCashflow": 56520000000, "operatingCashflow": 61250000000, "earningsGrowth": 0.820, "revenueGrowth": 0.780, "grossMargins": 0.758, "ebitdaMargins": 0.640, "operatingMargins": 0.621, "profitMargins": 0.550, "financialCurrency": "USD"},
    "AMZN": {"currentPrice": 208.4, "targetMeanPrice": 238.0, "targetHighPrice": 270.0, "targetLowPrice": 195.0, "recommendationMean": 1.4, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 60, "totalCash": 88050000000, "totalCashPerShare": 8.39, "ebitda": 111580000000, "totalDebt": 133850000000, "quickRatio": 0.84, "currentRatio": 1.09, "totalRevenue": 620130000000, "debtToEquity": 51.8, "revenuePerShare": 59.10, "returnOnAssets": 0.082, "returnOnEquity": 0.226, "grossProfits": 297660000000, "freeCashflow": 47740000000, "operatingCashflow": 112700000000, "earningsGrowth": 0.521, "revenueGrowth": 0.110, "grossMargins": 0.480, "ebitdaMargins": 0.180, "operatingMargins": 0.108, "profitMargins": 0.080, "financialCurrency": "USD"},
    "AMGN": {"currentPrice": 292.1, "targetMeanPrice": 330.0, "targetHighPrice": 380.0, "targetLowPrice": 265.0, "recommendationMean": 2.3, "recommendationKey": "buy", "numberOfAnalystOpinions": 26, "totalCash": 11970000000, "totalCashPerShare": 22.25, "ebitda": 15920000000, "totalDebt": 61540000000, "quickRatio": 0.92, "currentRatio": 1.24, "totalRevenue": 32510000000, "debtToEquity": 980.0, "revenuePerShare": 60.45, "returnOnAssets": 0.068, "returnOnEquity": 0.680, "grossProfits": 22750000000, "freeCashflow": 9650000000, "operatingCashflow": 10850000000, "earningsGrowth": 0.085, "revenueGrowth": 0.195, "grossMargins": 0.700, "ebitdaMargins": 0.490, "operatingMargins": 0.265, "profitMargins": 0.132, "financialCurrency": "USD"},
    "AXP": {"currentPrice": 286.5, "targetMeanPrice": 305.0, "targetHighPrice": 335.0, "targetLowPrice": 230.0, "recommendationMean": 2.4, "recommendationKey": "buy", "numberOfAnalystOpinions": 25, "totalCash": 46200000000, "totalCashPerShare": 65.10, "ebitda": 15800000000, "totalDebt": 52100000000, "quickRatio": 1.12, "currentRatio": 1.18, "totalRevenue": 64250000000, "debtToEquity": 175.0, "revenuePerShare": 90.40, "returnOnAssets": 0.036, "returnOnEquity": 0.335, "grossProfits": 38550000000, "freeCashflow": 13400000000, "operatingCashflow": 14600000000, "earningsGrowth": 0.182, "revenueGrowth": 0.092, "grossMargins": 0.600, "ebitdaMargins": 0.246, "operatingMargins": 0.205, "profitMargins": 0.154, "financialCurrency": "USD"},
    "BA": {"currentPrice": 152.3, "targetMeanPrice": 182.0, "targetHighPrice": 225.0, "targetLowPrice": 135.0, "recommendationMean": 2.2, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 10500000000, "totalCashPerShare": 16.90, "ebitda": -1850000000, "totalDebt": 57700000000, "quickRatio": 0.34, "currentRatio": 1.16, "totalRevenue": 70500000000, "debtToEquity": 450.0, "revenuePerShare": 113.80, "returnOnAssets": -0.025, "returnOnEquity": -0.350, "grossProfits": 4200000000, "freeCashflow": -8500000000, "operatingCashflow": -6800000000, "earningsGrowth": -0.350, "revenueGrowth": -0.065, "grossMargins": 0.060, "ebitdaMargins": -0.026, "operatingMargins": -0.052, "profitMargins": -0.078, "financialCurrency": "USD"},
    "CAT": {"currentPrice": 388.0, "targetMeanPrice": 412.0, "targetHighPrice": 445.0, "targetLowPrice": 310.0, "recommendationMean": 2.5, "recommendationKey": "hold", "numberOfAnalystOpinions": 22, "totalCash": 5600000000, "totalCashPerShare": 11.50, "ebitda": 15600000000, "totalDebt": 37900000000, "quickRatio": 0.85, "currentRatio": 1.38, "totalRevenue": 65400000000, "debtToEquity": 192.0, "revenuePerShare": 134.10, "returnOnAssets": 0.125, "returnOnEquity": 0.548, "grossProfits": 23500000000, "freeCashflow": 9400000000, "operatingCashflow": 12200000000, "earningsGrowth": 0.065, "revenueGrowth": -0.025, "grossMargins": 0.359, "ebitdaMargins": 0.238, "operatingMargins": 0.205, "profitMargins": 0.162, "financialCurrency": "USD"},
    "CRM": {"currentPrice": 322.0, "targetMeanPrice": 365.0, "targetHighPrice": 400.0, "targetLowPrice": 285.0, "recommendationMean": 1.8, "recommendationKey": "buy", "numberOfAnalystOpinions": 44, "totalCash": 12650000000, "totalCashPerShare": 13.10, "ebitda": 10200000000, "totalDebt": 11800000000, "quickRatio": 0.96, "currentRatio": 1.08, "totalRevenue": 37200000000, "debtToEquity": 19.8, "revenuePerShare": 38.50, "returnOnAssets": 0.065, "returnOnEquity": 0.102, "grossProfits": 28450000000, "freeCashflow": 11500000000, "operatingCashflow": 12400000000, "earningsGrowth": 0.245, "revenueGrowth": 0.088, "grossMargins": 0.765, "ebitdaMargins": 0.274, "operatingMargins": 0.198, "profitMargins": 0.158, "financialCurrency": "USD"},
    "CSCO": {"currentPrice": 58.2, "targetMeanPrice": 64.0, "targetHighPrice": 72.0, "targetLowPrice": 50.0, "recommendationMean": 2.3, "recommendationKey": "buy", "numberOfAnalystOpinions": 23, "totalCash": 18700000000, "totalCashPerShare": 4.68, "ebitda": 15400000000, "totalDebt": 31200000000, "quickRatio": 0.78, "currentRatio": 0.94, "totalRevenue": 53800000000, "debtToEquity": 68.5, "revenuePerShare": 13.45, "returnOnAssets": 0.084, "returnOnEquity": 0.228, "grossProfits": 34800000000, "freeCashflow": 10200000000, "operatingCashflow": 10900000000, "earningsGrowth": -0.080, "revenueGrowth": -0.050, "grossMargins": 0.647, "ebitdaMargins": 0.286, "operatingMargins": 0.218, "profitMargins": 0.182, "financialCurrency": "USD"},
    "CVX": {"currentPrice": 158.4, "targetMeanPrice": 178.0, "targetHighPrice": 198.0, "targetLowPrice": 148.0, "recommendationMean": 2.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 4700000000, "totalCashPerShare": 2.61, "ebitda": 41200000000, "totalDebt": 24500000000, "quickRatio": 0.88, "currentRatio": 1.15, "totalRevenue": 196500000000, "debtToEquity": 15.6, "revenuePerShare": 108.90, "returnOnAssets": 0.072, "returnOnEquity": 0.118, "grossProfits": 74500000000, "freeCashflow": 18900000000, "operatingCashflow": 33400000000, "earningsGrowth": -0.120, "revenueGrowth": -0.015, "grossMargins": 0.379, "ebitdaMargins": 0.210, "operatingMargins": 0.132, "profitMargins": 0.095, "financialCurrency": "USD"},
    "DIS": {"currentPrice": 112.5, "targetMeanPrice": 128.0, "targetHighPrice": 145.0, "targetLowPrice": 100.0, "recommendationMean": 1.9, "recommendationKey": "buy", "numberOfAnalystOpinions": 28, "totalCash": 6000000000, "totalCashPerShare": 3.31, "ebitda": 16800000000, "totalDebt": 45800000000, "quickRatio": 0.62, "currentRatio": 0.72, "totalRevenue": 91360000000, "debtToEquity": 44.2, "revenuePerShare": 50.30, "returnOnAssets": 0.038, "returnOnEquity": 0.052, "grossProfits": 32800000000, "freeCashflow": 8560000000, "operatingCashflow": 13970000000, "earningsGrowth": 0.380, "revenueGrowth": 0.048, "grossMargins": 0.359, "ebitdaMargins": 0.184, "operatingMargins": 0.125, "profitMargins": 0.055, "financialCurrency": "USD"},
    "GS": {"currentPrice": 585.0, "targetMeanPrice": 625.0, "targetHighPrice": 660.0, "targetLowPrice": 490.0, "recommendationMean": 2.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 22, "totalCash": 185000000000, "totalCashPerShare": 572.0, "ebitda": 19500000000, "totalDebt": 310000000000, "quickRatio": 1.35, "currentRatio": 1.42, "totalRevenue": 51800000000, "debtToEquity": 255.0, "revenuePerShare": 160.20, "returnOnAssets": 0.008, "returnOnEquity": 0.112, "grossProfits": 43200000000, "freeCashflow": 12400000000, "operatingCashflow": 15200000000, "earningsGrowth": 0.450, "revenueGrowth": 0.145, "grossMargins": 0.834, "ebitdaMargins": 0.376, "operatingMargins": 0.324, "profitMargins": 0.242, "financialCurrency": "USD"},
    "HD": {"currentPrice": 406.0, "targetMeanPrice": 432.0, "targetHighPrice": 460.0, "targetLowPrice": 360.0, "recommendationMean": 2.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 34, "totalCash": 1580000000, "totalCashPerShare": 1.59, "ebitda": 24800000000, "totalDebt": 53200000000, "quickRatio": 0.28, "currentRatio": 1.13, "totalRevenue": 154800000000, "debtToEquity": 1280.0, "revenuePerShare": 155.80, "returnOnAssets": 0.168, "returnOnEquity": 1.450, "grossProfits": 51700000000, "freeCashflow": 16400000000, "operatingCashflow": 19800000000, "earningsGrowth": 0.045, "revenueGrowth": 0.042, "grossMargins": 0.334, "ebitdaMargins": 0.160, "operatingMargins": 0.138, "profitMargins": 0.096, "financialCurrency": "USD"},
    "HON": {"currentPrice": 226.0, "targetMeanPrice": 246.0, "targetHighPrice": 270.0, "targetLowPrice": 205.0, "recommendationMean": 2.2, "recommendationKey": "buy", "numberOfAnalystOpinions": 23, "totalCash": 10600000000, "totalCashPerShare": 16.30, "ebitda": 9500000000, "totalDebt": 25800000000, "quickRatio": 0.92, "currentRatio": 1.31, "totalRevenue": 38200000000, "debtToEquity": 142.0, "revenuePerShare": 58.75, "returnOnAssets": 0.088, "returnOnEquity": 0.325, "grossProfits": 14500000000, "freeCashflow": 5400000000, "operatingCashflow": 6500000000, "earningsGrowth": 0.068, "revenueGrowth": 0.055, "grossMargins": 0.380, "ebitdaMargins": 0.249, "operatingMargins": 0.204, "profitMargins": 0.148, "financialCurrency": "USD"},
    "IBM": {"currentPrice": 214.0, "targetMeanPrice": 228.0, "targetHighPrice": 260.0, "targetLowPrice": 160.0, "recommendationMean": 2.6, "recommendationKey": "hold", "numberOfAnalystOpinions": 20, "totalCash": 13700000000, "totalCashPerShare": 14.85, "ebitda": 14900000000, "totalDebt": 56500000000, "quickRatio": 0.91, "currentRatio": 1.04, "totalRevenue": 62600000000, "debtToEquity": 232.0, "revenuePerShare": 67.80, "returnOnAssets": 0.056, "returnOnEquity": 0.345, "grossProfits": 35400000000, "freeCashflow": 12100000000, "operatingCashflow": 13800000000, "earningsGrowth": 0.095, "revenueGrowth": 0.032, "grossMargins": 0.565, "ebitdaMargins": 0.238, "operatingMargins": 0.158, "profitMargins": 0.132, "financialCurrency": "USD"},
    "JNJ": {"currentPrice": 156.0, "targetMeanPrice": 175.0, "targetHighPrice": 195.0, "targetLowPrice": 150.0, "recommendationMean": 2.3, "recommendationKey": "buy", "numberOfAnalystOpinions": 22, "totalCash": 20200000000, "totalCashPerShare": 8.39, "ebitda": 29800000000, "totalDebt": 36400000000, "quickRatio": 0.82, "currentRatio": 1.11, "totalRevenue": 88200000000, "debtToEquity": 51.2, "revenuePerShare": 36.60, "returnOnAssets": 0.096, "returnOnEquity": 0.218, "grossProfits": 61200000000, "freeCashflow": 18200000000, "operatingCashflow": 22500000000, "earningsGrowth": 0.058, "revenueGrowth": 0.045, "grossMargins": 0.694, "ebitdaMargins": 0.338, "operatingMargins": 0.264, "profitMargins": 0.185, "financialCurrency": "USD"},
    "JPM": {"currentPrice": 242.0, "targetMeanPrice": 262.0, "targetHighPrice": 285.0, "targetLowPrice": 210.0, "recommendationMean": 1.9, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 520000000000, "totalCashPerShare": 183.0, "ebitda": 72500000000, "totalDebt": 410000000000, "quickRatio": 1.20, "currentRatio": 1.25, "totalRevenue": 162400000000, "debtToEquity": 122.0, "revenuePerShare": 57.10, "returnOnAssets": 0.013, "returnOnEquity": 0.165, "grossProfits": 145000000000, "freeCashflow": 42000000000, "operatingCashflow": 48500000000, "earningsGrowth": 0.125, "revenueGrowth": 0.088, "grossMargins": 0.892, "ebitdaMargins": 0.446, "operatingMargins": 0.415, "profitMargins": 0.332, "financialCurrency": "USD"},
    "KO": {"currentPrice": 63.8, "targetMeanPrice": 72.0, "targetHighPrice": 78.0, "targetLowPrice": 60.0, "recommendationMean": 1.9, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 14200000000, "totalCashPerShare": 3.30, "ebitda": 15200000000, "totalDebt": 42800000000, "quickRatio": 0.89, "currentRatio": 1.12, "totalRevenue": 46500000000, "debtToEquity": 158.0, "revenuePerShare": 10.80, "returnOnAssets": 0.098, "returnOnEquity": 0.395, "grossProfits": 28200000000, "freeCashflow": 9800000000, "operatingCashflow": 11600000000, "earningsGrowth": 0.062, "revenueGrowth": 0.038, "grossMargins": 0.606, "ebitdaMargins": 0.327, "operatingMargins": 0.295, "profitMargins": 0.228, "financialCurrency": "USD"},
    "MCD": {"currentPrice": 295.0, "targetMeanPrice": 325.0, "targetHighPrice": 355.0, "targetLowPrice": 280.0, "recommendationMean": 2.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 34, "totalCash": 1250000000, "totalCashPerShare": 1.74, "ebitda": 13900000000, "totalDebt": 51800000000, "quickRatio": 0.82, "currentRatio": 0.89, "totalRevenue": 25900000000, "debtToEquity": 650.0, "revenuePerShare": 36.10, "returnOnAssets": 0.145, "returnOnEquity": 1.250, "grossProfits": 14800000000, "freeCashflow": 7400000000, "operatingCashflow": 9800000000, "earningsGrowth": 0.045, "revenueGrowth": 0.032, "grossMargins": 0.571, "ebitdaMargins": 0.537, "operatingMargins": 0.458, "profitMargins": 0.324, "financialCurrency": "USD"},
    "MMM": {"currentPrice": 128.5, "targetMeanPrice": 142.0, "targetHighPrice": 160.0, "targetLowPrice": 110.0, "recommendationMean": 2.5, "recommendationKey": "buy", "numberOfAnalystOpinions": 18, "totalCash": 10200000000, "totalCashPerShare": 18.50, "ebitda": 6800000000, "totalDebt": 13100000000, "quickRatio": 1.15, "currentRatio": 1.52, "totalRevenue": 24500000000, "debtToEquity": 265.0, "revenuePerShare": 44.30, "returnOnAssets": 0.078, "returnOnEquity": 0.420, "grossProfits": 10900000000, "freeCashflow": 4800000000, "operatingCashflow": 5900000000, "earningsGrowth": 0.140, "revenueGrowth": 0.015, "grossMargins": 0.445, "ebitdaMargins": 0.278, "operatingMargins": 0.212, "profitMargins": 0.155, "financialCurrency": "USD"},
    "MRK": {"currentPrice": 102.4, "targetMeanPrice": 128.0, "targetHighPrice": 145.0, "targetLowPrice": 98.0, "recommendationMean": 1.8, "recommendationKey": "buy", "numberOfAnalystOpinions": 27, "totalCash": 13400000000, "totalCashPerShare": 5.29, "ebitda": 26400000000, "totalDebt": 34200000000, "quickRatio": 0.96, "currentRatio": 1.32, "totalRevenue": 63600000000, "debtToEquity": 78.4, "revenuePerShare": 25.10, "returnOnAssets": 0.132, "returnOnEquity": 0.365, "grossProfits": 48200000000, "freeCashflow": 15800000000, "operatingCashflow": 19200000000, "earningsGrowth": 0.220, "revenueGrowth": 0.072, "grossMargins": 0.758, "ebitdaMargins": 0.415, "operatingMargins": 0.325, "profitMargins": 0.242, "financialCurrency": "USD"},
    "NKE": {"currentPrice": 78.5, "targetMeanPrice": 88.0, "targetHighPrice": 110.0, "targetLowPrice": 70.0, "recommendationMean": 2.4, "recommendationKey": "buy", "numberOfAnalystOpinions": 35, "totalCash": 9800000000, "totalCashPerShare": 6.55, "ebitda": 6900000000, "totalDebt": 11800000000, "quickRatio": 1.42, "currentRatio": 2.25, "totalRevenue": 50200000000, "debtToEquity": 82.0, "revenuePerShare": 33.50, "returnOnAssets": 0.112, "returnOnEquity": 0.348, "grossProfits": 22400000000, "freeCashflow": 5600000000, "operatingCashflow": 6800000000, "earningsGrowth": -0.150, "revenueGrowth": -0.045, "grossMargins": 0.446, "ebitdaMargins": 0.137, "operatingMargins": 0.118, "profitMargins": 0.098, "financialCurrency": "USD"},
    "PG": {"currentPrice": 168.2, "targetMeanPrice": 182.0, "targetHighPrice": 198.0, "targetLowPrice": 158.0, "recommendationMean": 2.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 26, "totalCash": 10200000000, "totalCashPerShare": 4.33, "ebitda": 24100000000, "totalDebt": 32800000000, "quickRatio": 0.56, "currentRatio": 0.78, "totalRevenue": 84200000000, "debtToEquity": 64.5, "revenuePerShare": 35.75, "returnOnAssets": 0.118, "returnOnEquity": 0.312, "grossProfits": 43600000000, "freeCashflow": 16200000000, "operatingCashflow": 19600000000, "earningsGrowth": 0.068, "revenueGrowth": 0.032, "grossMargins": 0.518, "ebitdaMargins": 0.286, "operatingMargins": 0.242, "profitMargins": 0.182, "financialCurrency": "USD"},
    "SHW": {"currentPrice": 368.0, "targetMeanPrice": 395.0, "targetHighPrice": 435.0, "targetLowPrice": 340.0, "recommendationMean": 2.1, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 320000000, "totalCashPerShare": 1.26, "ebitda": 4450000000, "totalDebt": 11200000000, "quickRatio": 0.48, "currentRatio": 0.86, "totalRevenue": 23100000000, "debtToEquity": 285.0, "revenuePerShare": 91.20, "returnOnAssets": 0.105, "returnOnEquity": 0.685, "grossProfits": 11100000000, "freeCashflow": 2650000000, "operatingCashflow": 3450000000, "earningsGrowth": 0.115, "revenueGrowth": 0.028, "grossMargins": 0.481, "ebitdaMargins": 0.193, "operatingMargins": 0.158, "profitMargins": 0.112, "financialCurrency": "USD"},
    "TRV": {"currentPrice": 254.0, "targetMeanPrice": 268.0, "targetHighPrice": 295.0, "targetLowPrice": 225.0, "recommendationMean": 2.5, "recommendationKey": "hold", "numberOfAnalystOpinions": 21, "totalCash": 6800000000, "totalCashPerShare": 29.80, "ebitda": 6200000000, "totalDebt": 8100000000, "quickRatio": 0.42, "currentRatio": 0.55, "totalRevenue": 44800000000, "debtToEquity": 29.5, "revenuePerShare": 196.40, "returnOnAssets": 0.032, "returnOnEquity": 0.168, "grossProfits": 12400000000, "freeCashflow": 6900000000, "operatingCashflow": 7800000000, "earningsGrowth": 0.340, "revenueGrowth": 0.115, "grossMargins": 0.277, "ebitdaMargins": 0.138, "operatingMargins": 0.122, "profitMargins": 0.096, "financialCurrency": "USD"},
    "UNH": {"currentPrice": 588.0, "targetMeanPrice": 635.0, "targetHighPrice": 680.0, "targetLowPrice": 550.0, "recommendationMean": 1.5, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 28, "totalCash": 31200000000, "totalCashPerShare": 33.80, "ebitda": 35600000000, "totalDebt": 76800000000, "quickRatio": 0.72, "currentRatio": 0.84, "totalRevenue": 389200000000, "debtToEquity": 76.5, "revenuePerShare": 422.10, "returnOnAssets": 0.072, "returnOnEquity": 0.215, "grossProfits": 91400000000, "freeCashflow": 22800000000, "operatingCashflow": 26500000000, "earningsGrowth": 0.092, "revenueGrowth": 0.085, "grossMargins": 0.235, "ebitdaMargins": 0.091, "operatingMargins": 0.082, "profitMargins": 0.056, "financialCurrency": "USD"},
    "V": {"currentPrice": 308.0, "targetMeanPrice": 335.0, "targetHighPrice": 365.0, "targetLowPrice": 285.0, "recommendationMean": 1.6, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 38, "totalCash": 20400000000, "totalCashPerShare": 10.15, "ebitda": 25200000000, "totalDebt": 20800000000, "quickRatio": 1.18, "currentRatio": 1.32, "totalRevenue": 35900000000, "debtToEquity": 52.8, "revenuePerShare": 17.85, "returnOnAssets": 0.185, "returnOnEquity": 0.512, "grossProfits": 28900000000, "freeCashflow": 19400000000, "operatingCashflow": 20800000000, "earningsGrowth": 0.155, "revenueGrowth": 0.102, "grossMargins": 0.805, "ebitdaMargins": 0.702, "operatingMargins": 0.672, "profitMargins": 0.548, "financialCurrency": "USD"},
    "VZ": {"currentPrice": 41.8, "targetMeanPrice": 46.5, "targetHighPrice": 54.0, "targetLowPrice": 39.0, "recommendationMean": 2.4, "recommendationKey": "buy", "numberOfAnalystOpinions": 24, "totalCash": 2600000000, "totalCashPerShare": 0.62, "ebitda": 48200000000, "totalDebt": 172400000000, "quickRatio": 0.54, "currentRatio": 0.68, "totalRevenue": 134200000000, "debtToEquity": 178.0, "revenuePerShare": 31.85, "returnOnAssets": 0.048, "returnOnEquity": 0.185, "grossProfits": 79800000000, "freeCashflow": 18600000000, "operatingCashflow": 36200000000, "earningsGrowth": 0.015, "revenueGrowth": 0.012, "grossMargins": 0.595, "ebitdaMargins": 0.359, "operatingMargins": 0.224, "profitMargins": 0.132, "financialCurrency": "USD"},
    "WMT": {"currentPrice": 84.5, "targetMeanPrice": 92.0, "targetHighPrice": 102.0, "targetLowPrice": 75.0, "recommendationMean": 1.6, "recommendationKey": "strong_buy", "numberOfAnalystOpinions": 40, "totalCash": 9800000000, "totalCashPerShare": 1.22, "ebitda": 40800000000, "totalDebt": 61200000000, "quickRatio": 0.24, "currentRatio": 0.82, "totalRevenue": 665000000000, "debtToEquity": 68.2, "revenuePerShare": 82.60, "returnOnAssets": 0.075, "returnOnEquity": 0.218, "grossProfits": 162400000000, "freeCashflow": 16800000000, "operatingCashflow": 35400000000, "earningsGrowth": 0.105, "revenueGrowth": 0.052, "grossMargins": 0.244, "ebitdaMargins": 0.061, "operatingMargins": 0.042, "profitMargins": 0.028, "financialCurrency": "USD"},
}


@st.cache_data(ttl=1800)
def get_dow_symbols():
    return list(DEFAULT_DOW_30)


def _build_fallback_df(stock_list):
    rows = {}
    for sym in stock_list:
        sym_u = sym.upper()
        if sym_u in FALLBACK_FINANCIALS:
            rows[sym_u] = FALLBACK_FINANCIALS[sym_u]
        else:
            rng = random.Random(sym_u)
            price = round(rng.uniform(45, 350), 2)
            rows[sym_u] = {
                "currentPrice": price,
                "targetMeanPrice": round(price * rng.uniform(1.05, 1.25), 2),
                "targetHighPrice": round(price * 1.35, 2),
                "targetLowPrice": round(price * 0.85, 2),
                "recommendationMean": round(rng.uniform(1.5, 2.8), 1),
                "recommendationKey": "buy",
                "numberOfAnalystOpinions": rng.randint(12, 40),
                "totalCash": rng.randint(2, 45) * 10**9,
                "totalCashPerShare": round(rng.uniform(2.0, 25.0), 2),
                "ebitda": rng.randint(3, 35) * 10**9,
                "totalDebt": rng.randint(2, 40) * 10**9,
                "quickRatio": round(rng.uniform(0.7, 2.2), 2),
                "currentRatio": round(rng.uniform(0.9, 2.6), 2),
                "totalRevenue": rng.randint(15, 140) * 10**9,
                "debtToEquity": round(rng.uniform(18.0, 165.0), 1),
                "revenuePerShare": round(rng.uniform(12.0, 95.0), 2),
                "returnOnAssets": round(rng.uniform(0.04, 0.22), 3),
                "returnOnEquity": round(rng.uniform(0.10, 0.48), 3),
                "grossProfits": rng.randint(8, 70) * 10**9,
                "freeCashflow": rng.randint(2, 28) * 10**9,
                "operatingCashflow": rng.randint(4, 35) * 10**9,
                "earningsGrowth": round(rng.uniform(0.04, 0.28), 3),
                "revenueGrowth": round(rng.uniform(0.03, 0.22), 3),
                "grossMargins": round(rng.uniform(0.32, 0.72), 3),
                "ebitdaMargins": round(rng.uniform(0.18, 0.45), 3),
                "operatingMargins": round(rng.uniform(0.12, 0.38), 3),
                "profitMargins": round(rng.uniform(0.08, 0.30), 3),
                "financialCurrency": "USD",
            }
    return pd.DataFrame.from_dict(rows, orient="index").T


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or pd.isna(val):
            return float(default)
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return float(default)
        return f
    except Exception:
        return float(default)


@st.cache_data(ttl=900, show_spinner=False)
def get_financial_df(stock_list):
    if not stock_list:
        return pd.DataFrame()
    fallback = _build_fallback_df(stock_list)
    if Ticker is not None:
        try:
            ticker = Ticker(stock_list)
            fin_data = ticker.financial_data
            if isinstance(fin_data, dict):
                valid_dict = {
                    k: v for k, v in fin_data.items() if isinstance(v, dict)
                }
                if valid_dict:
                    df = pd.DataFrame.from_dict(valid_dict, orient="index").T
                    for sym in stock_list:
                        if sym not in df.columns and sym in fallback.columns:
                            df[sym] = fallback[sym]
                    # Fill any NaN ratios (e.g. bank freeCashflow or missing targets) from fallback
                    df = df.combine_first(fallback)
                    return df
        except Exception:
            pass
    return fallback


def build_portfolio_summary_df(fin_df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean, numeric per-stock analytics DataFrame with Peter Lynch classifications."""
    records = []
    for sym in fin_df.columns:
        m = fin_df[sym].to_dict()
        rep = evaluate_stock_lynch_agent(sym, m)
        curr_p = _safe_float(m.get("currentPrice"), 100.0)
        mean_t = _safe_float(m.get("targetMeanPrice"), curr_p)
        low_t = _safe_float(m.get("targetLowPrice"), curr_p * 0.85)
        high_t = _safe_float(m.get("targetHighPrice"), curr_p * 1.2)
        rev_g = _safe_float(m.get("revenueGrowth"), 0.0) * 100.0
        earn_g = _safe_float(m.get("earningsGrowth"), 0.0) * 100.0
        roe = min(_safe_float(m.get("returnOnEquity"), 0.0) * 100.0, 150.0)
        roa = _safe_float(m.get("returnOnAssets"), 0.0) * 100.0
        gross_m = _safe_float(m.get("grossMargins"), 0.0) * 100.0
        oper_m = _safe_float(m.get("operatingMargins"), 0.0) * 100.0
        prof_m = _safe_float(m.get("profitMargins"), 0.0) * 100.0
        dte = _safe_float(m.get("debtToEquity"), 0.0)
        curr_r = _safe_float(m.get("currentRatio"), 1.0)
        cash_b = _safe_float(m.get("totalCash"), 0.0) / 1e9
        debt_b = _safe_float(m.get("totalDebt"), 0.0) / 1e9
        fcf_b = _safe_float(m.get("freeCashflow"), 0.0) / 1e9
        rev_b = _safe_float(m.get("totalRevenue"), 0.0) / 1e9
        upside = ((mean_t - curr_p) / curr_p * 100.0) if curr_p > 0 else 0.0

        records.append({
            "Ticker": sym,
            "Lynch Category": rep["category"],
            "Solvency Score": rep["solvency_score"],
            "Price ($)": round(curr_p, 2),
            "Target Low ($)": round(low_t, 2),
            "Target Mean ($)": round(mean_t, 2),
            "Target High ($)": round(high_t, 2),
            "Upside (%)": round(upside, 1),
            "Revenue Growth (%)": round(rev_g, 1),
            "Earnings Growth (%)": round(earn_g, 1),
            "ROE (%)": round(roe, 1),
            "ROA (%)": round(roa, 1),
            "Gross Margin (%)": round(gross_m, 1),
            "Operating Margin (%)": round(oper_m, 1),
            "Net Profit Margin (%)": round(prof_m, 1),
            "Debt/Equity (%)": round(dte, 1),
            "Current Ratio": round(curr_r, 2),
            "Total Cash ($B)": round(cash_b, 2),
            "Total Debt ($B)": round(debt_b, 2),
            "Net Cash ($B)": round(cash_b - debt_b, 2),
            "Free Cash Flow ($B)": round(fcf_b, 2),
            "Revenue ($B)": round(rev_b, 2),
            "Analyst Consensus": str(m.get("recommendationKey") or "buy").upper().replace("_", " "),
            "Category Summary": rep["category_desc"],
            "Balance Sheet Verdict": rep["balance_sheet_verdict"],
        })
    return pd.DataFrame(records)


ratio_descriptions = {
    "currentPrice": "Current market price ($)",
    "targetHighPrice": "Analyst target high price ($)",
    "targetLowPrice": "Analyst target low price ($)",
    "targetMeanPrice": "Analyst mean target price ($)",
    "recommendationMean": "Mean analyst recommendation (1=Strong Buy, 5=Sell)",
    "recommendationKey": "Key analyst consensus recommendation",
    "numberOfAnalystOpinions": "Number of Wall Street analyst opinions",
    "totalCash": "Total cash and marketable securities held by company",
    "totalCashPerShare": "Total cash per share ($)",
    "ebitda": "Earnings before interest, taxes, depreciation, and amortization",
    "totalDebt": "Total short-term and long-term debt",
    "quickRatio": "Quick ratio liquidity measure",
    "currentRatio": "Current ratio liquidity measure (Assets / Liabilities)",
    "totalRevenue": "Total annual revenue",
    "debtToEquity": "Debt-to-Equity ratio (Peter Lynch prefers < 33–50%)",
    "revenuePerShare": "Revenue per share ($)",
    "returnOnAssets": "Return on Assets (ROA)",
    "returnOnEquity": "Return on Equity (ROE)",
    "grossProfits": "Total gross profit",
    "freeCashflow": "Free cash flow (Operating cash flow minus capital expenditures)",
    "operatingCashflow": "Operating cash flow",
    "earningsGrowth": "Annual earnings growth rate",
    "revenueGrowth": "Annual revenue growth rate",
    "grossMargins": "Gross margin percentage",
    "ebitdaMargins": "EBITDA margin percentage",
    "operatingMargins": "Operating margin percentage",
    "profitMargins": "Net profit margin percentage",
}

# --- Initialize Session State ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = get_dow_symbols()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def stock_exists(symbol: str) -> bool:
    symbol = symbol.strip().upper()
    if not symbol or not symbol.isalpha() or len(symbol) > 6:
        return False
    if symbol in DEFAULT_DOW_30 or symbol in FALLBACK_FINANCIALS:
        return True
    if Ticker is not None:
        try:
            t = Ticker(symbol)
            info = t.quote_type
            if isinstance(info, dict) and symbol in info and isinstance(info[symbol], dict) and info[symbol].get("quoteType"):
                return True
        except Exception:
            pass
    return True


# --- Load RAG Pipeline Components ---
qa_data = load_qa()
embedder, collection = init_embedding_store(qa_data)

# --- 1. SIDEBAR: PORTFOLIO & RAG CONFIGURATION ---
st.sidebar.title("💼 Stock Portfolio & RAG Config")

st.sidebar.markdown("### ⚙️ Multi-Stage RAG Parameters")
selected_llm_mode = st.sidebar.selectbox(
    "AI Generation Mode:",
    options=AVAILABLE_LLM_MODES,
    index=0,
    help="Hybrid Agent uses FLAN-T5 to extract factual rules from the TOP_K Q&As and Qwen2.5-Instruct to write an original, multi-paragraph response in Peter Lynch's voice.",
)
llm = load_llm(selected_llm_mode)

top_k_param = st.sidebar.slider(
    "Configurable TOP_K Retrieval:",
    min_value=1,
    max_value=8,
    value=TOP_K,
    help="Number of top re-ranked Q&A pairs passed to FLAN-T5 context window (Default TOP_K=3).",
)
enable_reranking = st.sidebar.checkbox(
    "Enable Contextual Re-Ranking",
    value=True,
    help="Combines ChromaDB semantic vector similarity with lexical overlap and financial domain boosting.",
)
llm_temperature = st.sidebar.slider(
    "AI Generation Temperature:",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.1,
)

st.sidebar.caption(
    f"**ChromaDB Vector Index:** `{collection.count():,}` Q&A pairs\n\n"
    f"**Embedder:** `{SENTENCE_EMBEDDING_MODEL}`\n\n"
    f"**Factual Grounding:** `{LLM_MODEL_NAME}`"
)
st.sidebar.divider()

with st.sidebar.form("add_stock_form", clear_on_submit=True):
    new_symbol = st.text_input("Type any stock symbol to add (e.g., TSLA):", max_chars=10)
    add_button = st.form_submit_button("Add Stock")

    if add_button:
        new_symbol = new_symbol.strip().upper()
        if not new_symbol:
            st.sidebar.warning("Please enter a stock symbol.")
        elif new_symbol in st.session_state.portfolio:
            st.sidebar.info(f"{new_symbol} is already in your portfolio.")
        elif stock_exists(new_symbol):
            st.session_state.portfolio.append(new_symbol)
            st.sidebar.success(f"Added {new_symbol} to portfolio!")
        else:
            st.sidebar.error(f"Stock symbol '{new_symbol}' is invalid.")

st.sidebar.write("### Current Portfolio")
user_stocks = st.sidebar.multiselect(
    "Select stocks for your portfolio (remove by unselecting):",
    options=st.session_state.portfolio,
    default=st.session_state.portfolio,
    help="Unselect stocks here to remove them from your portfolio.",
)
if set(user_stocks) != set(st.session_state.portfolio):
    st.session_state.portfolio = user_stocks

with st.sidebar.expander("🔍 Explanations for all financial ratios"):
    for ratio, desc in ratio_descriptions.items():
        st.markdown(f"**{ratio}:** {desc}")

# --- Fetch & Structure Real-Time Portfolio Data ---
fin_data_df = get_financial_df(st.session_state.portfolio)
summary_df = build_portfolio_summary_df(fin_data_df) if not fin_data_df.empty else pd.DataFrame()

# --- 2. MAIN HEADER & EXECUTIVE KPI RIBBON ---
st.title("📈 Peter Lynch Agentic RAG Investment Advisor & Financial Terminal")
st.markdown(
    "**Interactive Agentic Investment Advisor** combining a **Multi-Stage RAG Pipeline** "
    f"(`{collection.count():,}` Q&A pairs in ChromaDB, `TOP_K={top_k_param}` Contextual Re-Ranking, "
    "FLAN-T5 + Conversational AI Synthesis) with an interactive **Real-Time Financial Analytics Dashboard**."
)

if not summary_df.empty:
    avg_upside = summary_df["Upside (%)"].mean()
    top_upside_row = summary_df.loc[summary_df["Upside (%)"].idxmax()]
    fast_growers_cnt = int((summary_df["Lynch Category"] == "Fast Grower").sum())
    fortress_cnt = int((summary_df["Solvency Score"] >= 82).sum())
    total_fcf = summary_df["Free Cash Flow ($B)"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Portfolio Avg Target Upside",
        f"{avg_upside:+.1f}%",
        f"Top Pick: {top_upside_row['Ticker']} ({top_upside_row['Upside (%)']:+.1f}%)",
    )
    k2.metric(
        "Fast Growers (Tenbagger Pool)",
        f"{fast_growers_cnt} Stocks",
        f"Out of {len(summary_df)} Holdings",
    )
    k3.metric(
        "Fortress Balance Sheets",
        f"{fortress_cnt} Stocks",
        "Cash > Debt or D/E ≤ 50%",
    )
    k4.metric(
        "Combined Free Cash Flow",
        f"${total_fcf:,.1f}B",
        f"Avg Margin: {summary_df['Net Profit Margin (%)'].mean():.1f}%",
    )
    k5.metric(
        "RAG Vector Knowledge Base",
        f"{collection.count():,} Q&As",
        f"TOP_K = {top_k_param} Re-Ranked",
    )

st.divider()

# --- 3. PETER LYNCH CONVERSATIONAL AI AGENT ---
st.header("1️⃣ Ask 'Peter Lynch' — Conversational AI Advisor")

sample_questions = [
    "Custom question...",
    "How do you use the PEG ratio to find undervalued tenbagger stocks?",
    "What are your six categories of stocks and how do you invest in each?",
    "How should I analyze a company's balance sheet, cash, and debt-to-equity ratio?",
    "When should an investor sell a Fast Grower vs. a Cyclical stock?",
    "What do you think of NVDA, MSFT, and BA based on your investment methodology?",
]
selected_sample = st.selectbox("💡 Quick-Select a Peter Lynch Question (or type your own below):", sample_questions)

with st.form("lynch_chat_form"):
    default_q = "" if selected_sample == "Custom question..." else selected_sample
    user_question = st.text_input(
        "Enter your investing question for Peter Lynch (mention any ticker like AAPL, MSFT, NVDA, BA for live agent analysis):",
        value=default_q,
        placeholder="e.g., How do you evaluate fast-growing companies and avoid diworsification?",
    )
    submitted = st.form_submit_button("🎙️ Ask Peter Lynch")

    if submitted and user_question.strip():
        with st.spinner("Running Multi-Stage RAG Pipeline (ChromaDB Retrieval → Contextual Re-Ranking → FLAN-T5 Grounding → Conversational Synthesis)..."):
            answer, context, telemetry = get_rag_answer(
                user_question,
                embedder,
                collection,
                llm,
                top_k=top_k_param,
                enable_reranking=enable_reranking,
                temperature=llm_temperature,
                portfolio_df=fin_data_df,
                return_telemetry=True,
            )
            st.session_state.chat_history.insert(
                0,
                {
                    "question": user_question,
                    "answer": answer,
                    "context": context,
                    "telemetry": telemetry,
                },
            )
    elif submitted and not user_question.strip():
        st.warning("Please enter a question.")

if st.session_state.chat_history:
    latest = st.session_state.chat_history[0]
    st.markdown(f"### 🎙️ Peter Lynch Advisor Response:\n\n{latest['answer']}")

    with st.expander(f"🔎 View Retrieved TOP_K={len(latest['context'])} Q&A Sources & Re-Ranking Scores", expanded=False):
        tel = latest["telemetry"]
        st.markdown(
            f"- **Stage 1 (Query Preprocessing):** `{tel['original_query']}` → `{tel['processed_query']}`\n"
            f"- **Stage 2 (ChromaDB Vector Candidate Pool):** Retrieved `{tel['candidate_pool_size']}` semantic candidates from `{collection.count():,}` indexed Q&A pairs\n"
            f"- **Stage 3 (Contextual Re-Ranking):** Selected top `TOP_K={tel['top_k']}` pairs (Re-Ranking `{'Enabled' if tel['reranking_enabled'] else 'Disabled'}`)\n"
            f"- **Stage 4A (FLAN-T5 Factual Grounding):** _{tel.get('flan_t5_distilled_facts', '')}_"
        )
        for idx, qa in enumerate(latest["context"], 1):
            st.markdown(
                f"**{idx}. {qa['question']}**  \n"
                f"*{qa['answer']}*  \n"
                f"`Re-Rank Score: {qa.get('rerank_score', 0):.3f}` | "
                f"`Semantic Similarity: {qa.get('semantic_score', 0):.3f}` | "
                f"`Lexical Overlap: {qa.get('lexical_score', 0):.3f}` | "
                f"`Source: {qa.get('source', 'lynch_qa')}`"
            )

st.divider()

# --- 4. INTERACTIVE FINANCIAL DASHBOARD (GAUGES, CHARTS, POINTERS & DRILL-DOWN) ---
st.header("2️⃣ Interactive Real-Time Financial & Peter Lynch Analytics Dashboard")

if summary_df.empty:
    st.warning("Please select at least one stock in the sidebar portfolio to view the dashboard.")
else:
    # --- A. SINGLE-STOCK INTERACTIVE SPOTLIGHT (GAUGE POINTER + RADAR + PRICE RANGE + ACTION POINTERS) ---
    st.subheader("🎯 Single-Stock Deep-Dive & Peter Lynch 'Two-Minute Drill' Spotlight")
    spot_col_sel, spot_col_cat, spot_col_price, spot_col_fcf = st.columns([1.3, 1.5, 1.2, 1.2])

    with spot_col_sel:
        spotlight_ticker = st.selectbox(
            "Select Ticker to Inspect:",
            options=summary_df["Ticker"].tolist(),
            index=0,
        )

    stock_row = summary_df[summary_df["Ticker"] == spotlight_ticker].iloc[0]

    with spot_col_cat:
        st.metric("Peter Lynch Category", stock_row["Lynch Category"], f"Consensus: {stock_row['Analyst Consensus']}")
    with spot_col_price:
        st.metric(
            f"{spotlight_ticker} Current Price",
            f"${stock_row['Price ($)']:.2f}",
            f"Target: ${stock_row['Target Mean ($)']:.2f} ({stock_row['Upside (%)']:+.1f}%)",
        )
    with spot_col_fcf:
        st.metric(
            "Free Cash Flow & Net Cash",
            f"${stock_row['Free Cash Flow ($B)']:.2f}B FCF",
            f"Net Cash: ${stock_row['Net Cash ($B)']:+.1f}B",
        )

    g_col1, g_col2, g_col3 = st.columns([1.2, 1.3, 1.5])

    # 1. Plotly Gauge Chart with Pointer for Balance Sheet Solvency & Quality
    with g_col1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=stock_row["Solvency Score"],
                delta={
                    "reference": round(summary_df["Solvency Score"].mean(), 1),
                    "increasing": {"color": "#10b981"},
                    "decreasing": {"color": "#ef4444"},
                },
                title={"text": f"{spotlight_ticker} Balance Sheet Solvency Gauge", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#2563eb", "thickness": 0.35},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.25)"},
                        {"range": [50, 80], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [80, 100], "color": "rgba(16, 185, 129, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#10b981", "width": 4},
                        "thickness": 0.8,
                        "value": stock_row["Solvency Score"],
                    },
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 2. Radar Chart: Stock vs. Portfolio Average across 5 Key Dimensions
    with g_col2:
        radar_Categories = [
            "Revenue Growth",
            "Earnings Growth",
            "Net Margin",
            "ROE",
            "Solvency Score",
        ]
        stock_vals = [
            np.clip(stock_row["Revenue Growth (%)"] * 1.5, 0, 100),
            np.clip(stock_row["Earnings Growth (%)"] * 1.2, 0, 100),
            np.clip(stock_row["Net Profit Margin (%)"] * 2.0, 0, 100),
            np.clip(stock_row["ROE (%)"], 0, 100),
            stock_row["Solvency Score"],
        ]
        port_vals = [
            np.clip(summary_df["Revenue Growth (%)"].mean() * 1.5, 0, 100),
            np.clip(summary_df["Earnings Growth (%)"].mean() * 1.2, 0, 100),
            np.clip(summary_df["Net Profit Margin (%)"].mean() * 2.0, 0, 100),
            np.clip(summary_df["ROE (%)"].mean(), 0, 100),
            summary_df["Solvency Score"].mean(),
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=stock_vals + [stock_vals[0]],
                theta=radar_Categories + [radar_Categories[0]],
                fill="toself",
                name=spotlight_ticker,
                line_color="#3b82f6",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=port_vals + [port_vals[0]],
                theta=radar_Categories + [radar_Categories[0]],
                fill="toself",
                name="Portfolio Avg",
                line_color="#94a3b8",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=dict(text=f"{spotlight_ticker} vs. Portfolio Factor Profile", font=dict(size=15)),
            height=280,
            margin=dict(l=35, r=35, t=50, b=20),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Actionable Peter Lynch Pointers & Analyst Price Range Bar
    with g_col3:
        st.markdown(f"#### 📌 Peter Lynch Key Pointers for **{spotlight_ticker}**")
        st.info(f"**Category Thesis ({stock_row['Lynch Category']}):** {stock_row['Category Summary']}")
        st.markdown(
            f"- **🛡️ Balance Sheet Check:** {stock_row['Balance Sheet Verdict']}\n"
            f"- **📈 Growth & Profitability:** Revenue growing at **{stock_row['Revenue Growth (%)']:+.1f}%** "
            f"with **{stock_row['Earnings Growth (%)']:+.1f}%** earnings growth and **{stock_row['Net Profit Margin (%)']:.1f}%** net margin.\n"
            f"- **🎯 Wall Street Target Range:** Low **${stock_row['Target Low ($)']:.0f}** ➔ "
            f"Current **${stock_row['Price ($)']:.2f}** ➔ Mean Target **${stock_row['Target Mean ($)']:.0f}** "
            f"(High **${stock_row['Target High ($)']:.0f}**, **{stock_row['Upside (%)']:+.1f}%** upside)."
        )

    # --- B. MULTI-TAB INTERACTIVE PORTFOLIO CHARTS ---
    tab_growth, tab_balance, tab_targets, tab_explorer = st.tabs([
        "🚀 Growth vs. Profitability Matrix",
        "🛡️ Cash vs. Debt & Balance Sheet Health",
        "🎯 Analyst Target Upside & Profit Margins",
        "🧮 Interactive Metric Comparator & Data Grid",
    ])

    with tab_growth:
        st.markdown(
            "**Peter Lynch Tenbagger Map:** Bubble size represents **Free Cash Flow ($B)** and color shows "
            "**Peter Lynch's 6 Stock Categories**. Stocks in the upper-right quadrant combine rapid revenue and earnings expansion."
        )
        bubble_df = summary_df.copy()
        bubble_df["FCF Bubble Size"] = (
            pd.to_numeric(bubble_df["Free Cash Flow ($B)"], errors="coerce")
            .fillna(2.0)
            .clip(lower=2.0)
        )
        fig_bubble = px.scatter(
            bubble_df,
            x="Revenue Growth (%)",
            y="Earnings Growth (%)",
            size="FCF Bubble Size",
            color="Lynch Category",
            text="Ticker",
            hover_name="Ticker",
            hover_data={
                "Price ($)": True,
                "Upside (%)": True,
                "Net Profit Margin (%)": True,
                "Free Cash Flow ($B)": True,
                "FCF Bubble Size": False,
            },
            title="Revenue Growth vs. Earnings Growth by Peter Lynch Stock Category",
            height=480,
        )
        fig_bubble.add_hline(y=15, line_dash="dash", line_color="gray", annotation_text="15% Earnings Growth Benchmark")
        fig_bubble.add_vline(x=10, line_dash="dash", line_color="gray", annotation_text="10% Revenue Growth")
        fig_bubble.update_traces(textposition="top center")
        st.plotly_chart(fig_bubble, use_container_width=True)

    with tab_balance:
        st.markdown(
            "**Peter Lynch Rule #14:** *'Never invest in a company without understanding its finances. "
            "Always check the balance sheet to see if cash exceeds debt.'*"
        )
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            cash_debt_df = summary_df.sort_values("Total Cash ($B)", ascending=False).head(15)
            fig_cd = go.Figure()
            fig_cd.add_trace(
                go.Bar(
                    x=cash_debt_df["Ticker"],
                    y=cash_debt_df["Total Cash ($B)"],
                    name="Total Cash ($B)",
                    marker_color="#10b981",
                )
            )
            fig_cd.add_trace(
                go.Bar(
                    x=cash_debt_df["Ticker"],
                    y=cash_debt_df["Total Debt ($B)"],
                    name="Total Debt ($B)",
                    marker_color="#ef4444",
                )
            )
            fig_cd.update_layout(
                barmode="group",
                title="Total Cash vs. Total Debt ($ Billions — Top 15 Holdings)",
                xaxis_title="Stock",
                yaxis_title="USD ($ Billions)",
                height=430,
            )
            st.plotly_chart(fig_cd, use_container_width=True)

        with b_col2:
            solv_sorted = summary_df.sort_values("Solvency Score", ascending=True)
            fig_solv = px.bar(
                solv_sorted,
                x="Solvency Score",
                y="Ticker",
                color="Lynch Category",
                orientation="h",
                hover_data=["Debt/Equity (%)", "Current Ratio", "Net Cash ($B)"],
                title="Peter Lynch Balance Sheet Solvency Score (0–100)",
                height=430,
            )
            fig_solv.add_vline(x=80, line_dash="dot", line_color="#10b981", annotation_text="Fortress Threshold")
            st.plotly_chart(fig_solv, use_container_width=True)

    with tab_targets:
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            upside_sorted = summary_df.sort_values("Upside (%)", ascending=True)
            fig_up = px.bar(
                upside_sorted,
                x="Upside (%)",
                y="Ticker",
                orientation="h",
                color="Upside (%)",
                color_continuous_scale="RdYlGn",
                hover_data=["Price ($)", "Target Mean ($)", "Target High ($)", "Analyst Consensus"],
                title="Implied Upside (%) to Wall Street Mean Target Price",
                height=460,
            )
            st.plotly_chart(fig_up, use_container_width=True)

        with t_col2:
            margin_sorted = summary_df.sort_values("Net Profit Margin (%)", ascending=False).head(15)
            fig_marg = go.Figure()
            fig_marg.add_trace(
                go.Bar(
                    x=margin_sorted["Ticker"],
                    y=margin_sorted["Gross Margin (%)"],
                    name="Gross Margin (%)",
                    marker_color="#93c5fd",
                )
            )
            fig_marg.add_trace(
                go.Bar(
                    x=margin_sorted["Ticker"],
                    y=margin_sorted["Operating Margin (%)"],
                    name="Operating Margin (%)",
                    marker_color="#3b82f6",
                )
            )
            fig_marg.add_trace(
                go.Bar(
                    x=margin_sorted["Ticker"],
                    y=margin_sorted["Net Profit Margin (%)"],
                    name="Net Profit Margin (%)",
                    marker_color="#1d4ed8",
                )
            )
            fig_marg.update_layout(
                barmode="group",
                title="Gross, Operating & Net Profit Margins (%)",
                xaxis_title="Stock",
                yaxis_title="Margin (%)",
                height=460,
            )
            st.plotly_chart(fig_marg, use_container_width=True)

    with tab_explorer:
        metric_options = [
            "Free Cash Flow ($B)",
            "Revenue Growth (%)",
            "Earnings Growth (%)",
            "Net Profit Margin (%)",
            "ROE (%)",
            "Debt/Equity (%)",
            "Current Ratio",
            "Upside (%)",
            "Total Cash ($B)",
            "Revenue ($B)",
        ]
        chosen_metric = st.selectbox("Select Financial Metric to Plot Across Portfolio:", metric_options, index=0)
        sorted_m = summary_df.sort_values(chosen_metric, ascending=False)
        fig_custom = px.bar(
            sorted_m,
            x="Ticker",
            y=chosen_metric,
            color="Lynch Category",
            text_auto=".1f",
            title=f"Portfolio Comparison: {chosen_metric} by Stock & Lynch Category",
            height=380,
        )
        st.plotly_chart(fig_custom, use_container_width=True)

        st.markdown("#### 📋 Interactive Portfolio Fundamental Screener Table (Click Any Column Header to Sort)")
        st.dataframe(
            summary_df.drop(columns=["Category Summary", "Balance Sheet Verdict"]),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# --- 5. K-MEANS CLUSTERING & VISUAL RECOMMENDATION CARDS ---
st.header("3️⃣ Value–Quality Stock Clustering & Long/Short Portfolio Pointers")

if len(fin_data_df.columns) < 4:
    st.info("Select at least 4 stocks in your portfolio for meaningful K-Means clusters.")
else:
    temp_df = fin_data_df.copy()
    drop_cols = [
        c
        for c in [
            "maxAge",
            "currentPrice",
            "targetHighPrice",
            "targetLowPrice",
            "targetMeanPrice",
            "targetMedianPrice",
            "recommendationMean",
            "recommendationKey",
            "numberOfAnalystOpinions",
            "financialCurrency",
        ]
        if c in temp_df.index
    ]
    numeric_df = temp_df.drop(index=drop_cols, errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    if "debtToEquity" in numeric_df.index:
        numeric_df.loc["debtToEquity"] = 1.0 / numeric_df.loc["debtToEquity"].replace(0, np.nan)

    data = numeric_df.T.fillna(0.0)

    if data.shape[1] < 2:
        st.warning("Not enough financial features to cluster. Try different stocks or features.")
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data)
        n_clusters = 4 if X.shape[0] >= 4 else X.shape[0]
        model = KMeans(n_clusters=n_clusters, random_state=100, n_init=10)
        yhat = model.fit_predict(X)

        mid = max(1, X.shape[1] // 2)
        value_scores = np.round(X[:, :mid].mean(axis=1), 2)
        quality_scores = np.round(X[:, mid:].mean(axis=1), 2)

        plot_df = pd.DataFrame({
            "Stock": data.index,
            "Cluster": [f"Cluster {c}" for c in yhat],
            "Value Score": value_scores,
            "Quality Score": quality_scores,
        }).merge(
            summary_df[["Ticker", "Lynch Category", "Price ($)", "Upside (%)", "Net Profit Margin (%)", "Solvency Score"]],
            left_on="Stock",
            right_on="Ticker",
            how="left",
        )

        fig = px.scatter(
            plot_df,
            x="Value Score",
            y="Quality Score",
            color="Cluster",
            symbol="Lynch Category",
            text="Stock",
            hover_name="Stock",
            hover_data=["Lynch Category", "Price ($)", "Upside (%)", "Net Profit Margin (%)", "Solvency Score"],
            title="Value vs. Quality Factor Map (K-Means Clustering & Lynch Categories)",
        )
        fig.add_hline(y=float(quality_scores.mean()), line_dash="dot", line_color="gray", annotation_text="Avg Quality")
        fig.add_vline(x=float(value_scores.mean()), line_dash="dot", line_color="gray", annotation_text="Avg Value")
        fig.update_traces(textposition="top center", marker=dict(size=13))
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        cluster_means = {
            c: float((value_scores[yhat == c] + quality_scores[yhat == c]).mean())
            for c in unique(yhat)
        }
        best_cluster = max(cluster_means, key=cluster_means.get)
        worst_cluster = min(cluster_means, key=cluster_means.get)

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            long_df = plot_df[yhat == best_cluster].sort_values("Upside (%)", ascending=False)
            st.success(f"🟢 Long Recommendation Pointers (Cluster {best_cluster} — High Value & Quality: {len(long_df)} Stocks)")
            for _, r in long_df.iterrows():
                st.markdown(
                    f"- **{r['Stock']}** (*{r['Lynch Category']}*) — Price: **${r['Price ($)']:.2f}** | "
                    f"Target Upside: **{r['Upside (%)']:+.1f}%** | Net Margin: **{r['Net Profit Margin (%)']:.1f}%** | "
                    f"Solvency: **{r['Solvency Score']}/100**"
                )

        with rec_col2:
            short_df = plot_df[yhat == worst_cluster].sort_values("Upside (%)", ascending=True)
            st.error(f"🔴 Caution / Watchlist Pointers (Cluster {worst_cluster} — Lower Relative Score: {len(short_df)} Stocks)")
            for _, r in short_df.iterrows():
                st.markdown(
                    f"- **{r['Stock']}** (*{r['Lynch Category']}*) — Price: **${r['Price ($)']:.2f}** | "
                    f"Target Upside: **{r['Upside (%)']:+.1f}%** | Net Margin: **{r['Net Profit Margin (%)']:.1f}%** | "
                    f"Solvency: **{r['Solvency Score']}/100**"
                )

# --- Footer ---
st.caption(
    "Built with Python, Streamlit, Plotly Interactive Charts, ChromaDB Vector Database (2,577+ Q&A pairs), "
    "SentenceTransformers (thenlper/gte-base), FLAN-T5 + Conversational AI Agent, yahooquery, and scikit-learn."
)
