# 필요한 라이브러리 임포트
import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
import ta
from ta.utils import dropna
import time
import requests
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import sqlite3
import schedule
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt
import pytz
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from threading import Lock
import numpy as np
import openai  # OpenAI 라이브러리 임포트

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 파일 핸들러 설정
file_handler = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=2)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

data_lock = Lock()  # 스레드 안전성을 위한 Lock 객체

class PortfolioAllocation(BaseModel):
    target_btc_ratio: float
    reason: str

def load_environment():
    load_dotenv()
    # 환경 변수 로드
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["UPBIT_ACCESS_KEY"] = os.getenv("UPBIT_ACCESS_KEY")
    os.environ["UPBIT_SECRET_KEY"] = os.getenv("UPBIT_SECRET_KEY")
    os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

def init_db():
    with sqlite3.connect('bitcoin_trades.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    decision TEXT,
                    percentage REAL,
                    reason TEXT,
                    btc_balance REAL,
                    krw_balance REAL,
                    btc_avg_buy_price REAL,
                    btc_krw_price REAL,
                    target_btc_ratio REAL,
                    current_btc_ratio REAL,
                    difference REAL,
                    executed_percentage REAL,
                    reflection TEXT)''')
        conn.commit()
    logger.info("데이터베이스 초기화 완료.")

def log_trade(**kwargs):
    try:
        with sqlite3.connect('bitcoin_trades.db') as conn:
            c = conn.cursor()
            # 뉴욕 시간대로 타임스탬프 생성
            ny_timezone = pytz.timezone("America/New_York")
            timestamp = datetime.now(ny_timezone).isoformat()
            kwargs['timestamp'] = timestamp

            columns = ', '.join(kwargs.keys())
            placeholders = ', '.join(['?'] * len(kwargs))
            sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            c.execute(sql, tuple(kwargs.values()))
            conn.commit()
        logger.info("거래 정보가 데이터베이스에 기록되었습니다.")
    except Exception as e:
        logger.error(f"거래 기록 오류: {e}")

def get_recent_trades(days=30):
    with sqlite3.connect('bitcoin_trades.db') as conn:
        c = conn.cursor()
        thirty_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (thirty_days_ago,))
        columns = [column[0] for column in c.description]
        data = c.fetchall()
        if not data:
            logger.info("최근 거래 내역이 없습니다.")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(data=data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    logger.info(f"최근 {len(df)}건의 거래 내역을 가져왔습니다.")
    return df

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    performance = (final_balance - initial_balance) / initial_balance * 100
    logger.info(f"최근 30일간의 거래 성과: {performance:.2f}%")
    return performance

def add_indicators(df):
    if df.empty:
        logger.warning("지표를 추가할 데이터가 없습니다.")
        return df
    try:
        # 볼린저 밴드
        indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # MFI
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14
        ).money_flow_index()

        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # 이동평균선
        df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_120'] = ta.trend.EMAIndicator(close=df['close'], window=120).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()

        logger.info("기술적 지표 추가 완료.")
    except Exception as e:
        logger.error(f"지표 추가 오류: {e}")
    return df

def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logger.info("공포와 탐욕 지수 데이터를 가져왔습니다.")
        return data['data'][0]
    except requests.RequestException as e:
        logger.error(f"공포와 탐욕 지수 데이터 가져오기 실패: {e}")
        return {"value": 50}

def get_bitcoin_news():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": "btc",
        "api_key": serpapi_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        news_results = data.get("news_results", [])
        headlines = [{"title": item.get("title", ""), "date": item.get("date", "")} for item in news_results]
        logger.info("비트코인 뉴스 데이터를 가져왔습니다.")
        return headlines[:10]
    except requests.RequestException as e:
        logger.error(f"뉴스 데이터 가져오기 오류: {e}")
        return []

def convert_timestamps_in_data(data):
    if isinstance(data, dict):
        return {k: convert_timestamps_in_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps_in_data(v) for v in data]
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.isoformat()
    else:
        return data

def ai_trading():
    load_environment()
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)

    # 전략 가져오기
    wonyyotti_strategy = get_combined_transcript()

    # 데이터 수집 함수 정의
    def fetch_balances():
        try:
            all_balances = upbit.get_balances()
            filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
            logger.info("잔고 데이터를 가져왔습니다.")
            return filtered_balances
        except Exception as e:
            logger.error(f"잔고 데이터 가져오기 오류: {e}")
            return []

    def fetch_orderbook():
        try:
            orderbook = pyupbit.get_orderbook("KRW-BTC")
            logger.info("호가창 데이터를 가져왔습니다.")
            return orderbook
        except Exception as e:
            logger.error(f"호가창 데이터 가져오기 오류: {e}")
            return {}

    def fetch_ohlcv(market, interval, count):
        try:
            df = pyupbit.get_ohlcv(market, interval=interval, count=count)
            df = dropna(df)
            df = add_indicators(df)
            logger.info(f"{market}의 {interval} 데이터 가져오기 완료.")
            return df
        except Exception as e:
            logger.error(f"{market}의 {interval} 데이터 가져오기 오류: {e}")
            return pd.DataFrame()

    def fetch_usd_ohlcv(timeframe, limit):
        try:
            exchange = ccxt.kraken()
            data = exchange.fetch_ohlcv('BTC/USD', timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = dropna(df)
            df = add_indicators(df)
            logger.info(f"USD-BTC의 {timeframe} 데이터 가져오기 완료.")
            return df
        except Exception as e:
            logger.error(f"USD-BTC의 {timeframe} 데이터 가져오기 오류: {e}")
            return pd.DataFrame()

    def fetch_fear_and_greed():
        return get_fear_and_greed_index()

    def fetch_news():
        return get_bitcoin_news()

    def fetch_transcript():
        return get_combined_transcript()

    def fetch_usd_krw_exchange_rate():
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            data = response.json()
            logger.info("USD/KRW 환율 데이터를 가져왔습니다.")
            return data['rates']['KRW']
        except Exception as e:
            logger.error(f"USD/KRW 환율 데이터 가져오기 오류: {e}")
            return None

    # 데이터 수집 실행
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_balances): 'balances',
            executor.submit(fetch_orderbook): 'orderbook',
            executor.submit(fetch_ohlcv, "KRW-BTC", "day", 200): 'daily_ohlcv',
            executor.submit(fetch_ohlcv, "KRW-BTC", "minute60", 200): 'hourly_ohlcv',
            executor.submit(fetch_usd_ohlcv, '1d', 200): 'daily_usd_ohlcv',
            executor.submit(fetch_usd_ohlcv, '1h', 200): 'hourly_usd_ohlcv',
            executor.submit(fetch_fear_and_greed): 'fear_greed',
            executor.submit(fetch_news): 'news',
            executor.submit(fetch_transcript): 'transcript',
            executor.submit(fetch_usd_krw_exchange_rate): 'usd_krw_rate',
        }

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                logger.info(f"{key} 데이터 수집 완료.")
            except Exception as e:
                logger.error(f"{key} 데이터 수집 중 오류 발생: {e}")
                results[key] = None

    # 수집된 데이터 활용
    filtered_balances = results.get('balances', [])
    orderbook = results.get('orderbook', {})
    df_daily_krw = results.get('daily_ohlcv', pd.DataFrame())
    df_hourly_krw = results.get('hourly_ohlcv', pd.DataFrame())
    df_daily_usd = results.get('daily_usd_ohlcv', pd.DataFrame())
    df_hourly_usd = results.get('hourly_usd_ohlcv', pd.DataFrame())
    fear_greed_index = results.get('fear_greed', {})
    news_headlines = results.get('news', [])
    youtube_transcript = results.get('transcript', "")
    usd_krw_rate = results.get('usd_krw_rate', None)

    # 필요한 컬럼만 선택
    def select_columns(df):
        if df.empty:
            return df
        return df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'mfi', 'ema_20', 'ema_50', 'ema_120', 'ema_200']]

    df_daily_krw = select_columns(df_daily_krw)
    df_hourly_krw = select_columns(df_hourly_krw)
    df_daily_usd = select_columns(df_daily_usd)
    df_hourly_usd = select_columns(df_hourly_usd)

    # 괴리율 계산
    if not df_hourly_usd.empty and usd_krw_rate:
        usd_price = df_hourly_usd['close'].iloc[-1]
        krw_price = pyupbit.get_current_price("KRW-BTC")
        if usd_price is not None and krw_price is not None:
            usd_price_in_krw = usd_price * usd_krw_rate
            premium = ((krw_price - usd_price_in_krw) / usd_price_in_krw) * 100
            logger.info(f"괴리율 계산 완료: {premium:.2f}%")
        else:
            premium = None
            logger.error("USD 가격 또는 KRW 가격이 유효하지 않습니다.")
    else:
        premium = None
        logger.error("USD OHLCV 데이터 또는 USD/KRW 환율이 유효하지 않습니다.")

    premium_formatted = f"{premium:.2f}" if premium is not None else "N/A"

    # 머신러닝 모델을 사용하여 예측 수행
    def prepare_ml_data():
        trades_df = get_recent_trades(days=30)
        if trades_df.empty:
            logger.warning("최근 거래 내역이 없습니다. 머신러닝 모델 학습을 건너뜁니다.")
            return None, None, None

        df_features = df_hourly_krw.copy()
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'timestamp'}, inplace=True)
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], utc=True)

        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], utc=True)
        merged_df = pd.merge_asof(trades_df.sort_values('timestamp'), df_features.sort_values('timestamp'), on='timestamp', direction='nearest')

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'mfi', 'ema_20', 'ema_50', 'ema_120', 'ema_200', 'fear_greed_score', 'premium']
        merged_df['fear_greed_score'] = fear_greed_index.get('value', 50)
        merged_df['premium'] = premium

        label_encoder = LabelEncoder()
        if 'decision' in merged_df.columns:
            merged_df['decision_label'] = label_encoder.fit_transform(merged_df['decision'])
        else:
            logger.error("'decision' 컬럼이 존재하지 않습니다.")
            return None, None, None

        merged_df.dropna(subset=feature_columns + ['decision_label'], inplace=True)

        X = merged_df[feature_columns]
        y = merged_df['decision_label']

        logger.info(f"머신러닝 모델 학습을 위한 데이터 준비 완료: X 샘플 수={len(X)}, y 샘플 수={len(y)}")
        return X, y, label_encoder

    X, y, label_encoder = prepare_ml_data()

    if X is not None and y is not None and len(X) > 2:
        num_samples = len(X)
        n_splits = min(5, max(2, num_samples - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"TimeSeriesSplit 설정: n_splits={n_splits}")

        def objective(trial):
            param = {
                'objective': 'multi:softprob',
                'num_class': len(set(y)),
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }

            accuracy_scores = []
            for train_index, test_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val, preds)
                accuracy_scores.append(accuracy)

            mean_accuracy = np.mean(accuracy_scores)
            logger.info(f"Trial {trial.number}의 평균 정확도: {mean_accuracy:.4f}")
            return -mean_accuracy

        study = optuna.create_study(direction='minimize')
        logger.info("Optuna 하이퍼파라미터 최적화 시작.")
        study.optimize(objective, n_trials=30)

        best_params = study.best_params
        logger.info(f"Optuna 최적화 완료. 최적 파라미터: {best_params}")

        model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X, y)
        logger.info("머신러닝 모델 학습 완료.")

        latest_data = df_hourly_krw.iloc[-1:].copy()
        latest_data['fear_greed_score'] = fear_greed_index.get('value', 50)
        latest_data['premium'] = premium
        latest_X = latest_data[feature_columns].fillna(0)

        ml_prediction = model.predict(latest_X)[0]
        ml_decision = label_encoder.inverse_transform([ml_prediction])[0]
        logger.info(f"머신러닝 모델 예측 결과: {ml_decision}")
    else:
        logger.info("머신러닝 모델을 학습하기에 충분한 데이터가 없습니다. OpenAI API의 결정을 기본값으로 사용합니다.")
        ml_decision = "hold"

    # 데이터 수집 후, 인덱스 재설정
    df_daily_krw.reset_index(inplace=True)
    df_hourly_krw.reset_index(inplace=True)
    df_daily_usd.reset_index(inplace=True)
    df_hourly_usd.reset_index(inplace=True)

    # DataFrames를 dict로 변환하고 Timestamp를 문자열로 변환
    df_daily_krw_data = convert_timestamps_in_data(df_daily_krw.to_dict(orient='records'))
    df_hourly_krw_data = convert_timestamps_in_data(df_hourly_krw.to_dict(orient='records'))
    df_daily_usd_data = convert_timestamps_in_data(df_daily_usd.to_dict(orient='records'))
    df_hourly_usd_data = convert_timestamps_in_data(df_hourly_usd.to_dict(orient='records'))

    # 기타 데이터들도 Timestamp를 문자열로 변환
    filtered_balances = convert_timestamps_in_data(filtered_balances)
    orderbook = convert_timestamps_in_data(orderbook)
    news_headlines = convert_timestamps_in_data(news_headlines)
    fear_greed_index = convert_timestamps_in_data(fear_greed_index)

    # OpenAI API를 사용하여 목표 비중 계산
    openai.api_key = os.getenv("OPENAI_API_KEY")

    recent_trades = get_recent_trades()
    trades_data = convert_timestamps_in_data(recent_trades.to_dict(orient='records'))

    current_market_data = {
        "fear_greed_index": fear_greed_index,
        "news_headlines": news_headlines,
        "orderbook": orderbook,
        "daily_ohlcv_krw": df_daily_krw_data,
        "hourly_ohlcv_krw": df_hourly_krw_data,
        "daily_ohlcv_usd": df_daily_usd_data,
        "hourly_ohlcv_usd": df_hourly_usd_data,
        "usd_krw_rate": usd_krw_rate,
        "krw_usd_premium": premium,
    }

    current_market_data = convert_timestamps_in_data(current_market_data)

    reflection = generate_reflection(recent_trades, current_market_data, wonyyotti_strategy)

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": f"""당신은 비트코인 투자 전문가입니다. 제공된 데이터를 분석하고, '워뇨띠'의 매매 전략을 참고하여 포트폴리오에서 비트코인의 **목표 비중(%)**을 결정하세요.

분석 시 고려해야 할 요소:
- USD 및 KRW 시장의 기술적 지표(RSI, MACD 등)
- USD와 KRW 간 환율
- USD 가격을 KRW로 환산한 값과 현재 KRW 가격 간의 괴리율
- 최근 뉴스 헤드라인과 그 영향
- 공포와 탐욕 지수의 의미
- 전체적인 시장 심리
- 최근 거래 성과 및 반성 내용

특히, 한국의 전설적인 투자자인 '워뇨띠'의 매매 방법을 항상 참고하여 현재 상황을 평가하고 거래 결정을 내려야 합니다. '워뇨띠'의 매매 방법은 다음과 같습니다:
{wonyyotti_strategy}

최근 거래 반성 내용:
{reflection}

비트코인의 목표 비중은 0에서 100 사이의 값이어야 합니다.

응답 형식:
{{
    "target_btc_ratio": 비트코인 목표 비중 (%),
    "reason": "결정한 이유"
}}

응답은 JSON 형식으로 제공해주세요."""
            },
            {
                "role": "user",
                "content": f"""현재 투자 현황: {json.dumps(filtered_balances)}
주문장: {json.dumps(orderbook)}
KRW 일간 OHLCV 지표 (30일): {json.dumps(df_daily_krw_data)}
KRW 시간별 OHLCV 지표 (24시간): {json.dumps(df_hourly_krw_data)}
USD 일간 OHLCV 지표 (30일): {json.dumps(df_daily_usd_data)}
USD 시간별 OHLCV 지표 (24시간): {json.dumps(df_hourly_usd_data)}
USD/KRW 환율: {usd_krw_rate}
KRW-USD 프리미엄 (%): {premium_formatted}
최근 뉴스 헤드라인: {json.dumps(news_headlines)}
공포와 탐욕 지수: {json.dumps(fear_greed_index)}"""
            }
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    try:
        result_content = response.choices[0].message.content
        result = json.loads(result_content)
        openai_target_btc_ratio = result['target_btc_ratio']
        openai_reason = result['reason']
        logger.info(f"OpenAI API 응답: {result}")
    except Exception as e:
        logger.error(f"OpenAI API 응답 파싱 오류: {e}")
        openai_target_btc_ratio = 0  # 기본값 설정
        openai_reason = "OpenAI API 응답 오류로 인해 비트코인 비중을 0으로 설정"

    # 머신러닝 모델과 OpenAI 모델의 결정을 결합하여 최종 결정
    time.sleep(1)
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")
    total_asset = krw_balance + btc_balance * current_btc_price
    current_btc_ratio = (btc_balance * current_btc_price) / total_asset * 100 if total_asset > 0 else 0

    # 머신러닝 모델과 OpenAI 모델의 목표 비중 결합
    if ml_decision == "buy":
        final_target_btc_ratio = max(openai_target_btc_ratio, current_btc_ratio + 5)  # 최소 5% 이상 증가
    elif ml_decision == "sell":
        final_target_btc_ratio = min(openai_target_btc_ratio, current_btc_ratio - 5)  # 최소 5% 이상 감소
    else:
        final_target_btc_ratio = (openai_target_btc_ratio + current_btc_ratio) / 2

    logger.info(f"최종 목표 비중: {final_target_btc_ratio:.2f}%, 현재 비중: {current_btc_ratio:.2f}%")

    # 목표 비중과 현재 비중의 차이 계산
    difference = final_target_btc_ratio - current_btc_ratio

    order_executed = False

    if difference > 0:
        # 매수 실행
        buy_amount_krw = total_asset * (difference / 100)
        if buy_amount_krw > 5000:
            logger.info(f"### 매수 주문 실행: {buy_amount_krw:.2f} KRW 상당의 BTC ###")
            try:
                order = upbit.buy_market_order("KRW-BTC", buy_amount_krw)
                if order:
                    order_executed = True
                    logger.info(f"매수 주문 성공: {order}")
            except Exception as e:
                logger.error(f"매수 주문 오류: {e}")
        else:
            logger.info("### 매수 주문 실패: KRW 금액 부족 ###")
    elif difference < 0:
        # 매도 실행
        sell_amount_btc = btc_balance * (-difference / current_btc_ratio)
        if sell_amount_btc * current_btc_price > 5000:
            logger.info(f"### 매도 주문 실행: {sell_amount_btc:.8f} BTC ###")
            try:
                order = upbit.sell_market_order("KRW-BTC", sell_amount_btc)
                if order:
                    order_executed = True
                    logger.info(f"매도 주문 성공: {order}")
            except Exception as e:
                logger.error(f"매도 주문 오류: {e}")
        else:
            logger.info("### 매도 주문 실패: BTC 금액 부족 ###")
    else:
        logger.info("### 포지션 유지 ###")

    # 거래 정보 로깅
    log_trade(
        decision=ml_decision,
        percentage=abs(difference) if order_executed else 0,
        reason=f"머신러닝 모델: {ml_decision}, OpenAI 결정: {openai_reason}",
        btc_balance=btc_balance,
        krw_balance=krw_balance,
        btc_avg_buy_price=btc_avg_buy_price,
        btc_krw_price=current_btc_price,
        target_btc_ratio=final_target_btc_ratio,
        current_btc_ratio=current_btc_ratio,
        difference=difference,
        executed_percentage=abs(difference) if order_executed else 0,
        reflection=reflection
    )

def generate_reflection(trades_df, current_market_data, wonyyotti_strategy):
    performance = calculate_performance(trades_df)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    trades_data = convert_timestamps_in_data(trades_df.to_dict(orient='records'))
    current_market_data = convert_timestamps_in_data(current_market_data)

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 AI 트레이딩 어시스턴트입니다. 최근 거래 성과와 현재 시장 상황을 분석하여 "
                    "미래 거래 결정을 개선하기 위한 인사이트를 제공합니다."
                )
            },
            {
                "role": "user",
                "content": f"""
최근 거래 데이터:
{json.dumps(trades_data)}

현재 시장 데이터:
{json.dumps(current_market_data)}

최근 7일간의 전체 성과: {performance:.2f}%

트레이딩 전략 참고(워뇨띠의 유튜브 매매 전략):
{wonyyotti_strategy}

이 데이터를 분석하고 다음을 제공해주세요:
1. 최근 거래 결정에 대한 간단한 반성
2. 잘된 점과 개선이 필요한 점
3. 향후 거래 결정을 위한 개선 방안 제시
4. 시장 데이터에서 관찰되는 패턴이나 트렌드

응답은 250단어 이내로 작성해주세요.
"""
            }
        ],
        max_tokens=500,
        temperature=0.7,
    )
    try:
        reflection_content = response['choices'][0]['message']['content']
        logger.info("거래 반성 내용 생성 완료.")
        return reflection_content
    except Exception as e:
        logger.error(f"거래 반성 내용 생성 오류: {e}")
        return "거래 반성 내용을 생성할 수 없습니다."

def get_combined_transcript():
    try:
        with open("strategy.txt", "r", encoding="utf-8") as f:
            combined_text = f.read()
        logger.info("전략 텍스트 파일 읽기 완료.")
        return combined_text
    except Exception as e:
        logger.error(f"전략 텍스트 파일 읽기 오류: {e}")
        return ""

def job():
    try:
        ai_trading()
    except Exception as e:
        logger.error(f"트레이딩 실행 중 오류 발생: {e}")

# def schedule_jobs():
#     ny_timezone = pytz.timezone("America/New_York")
#     schedule.every().day.at("00:00").do(job)
#     schedule.every().day.at("08:00").do(job)
#     schedule.every().day.at("16:00").do(job)

#     try:
#         while True:
#             schedule.run_pending()
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logger.info("스케줄 작업이 중단되었습니다.")

if __name__ == "__main__":
    init_db()
    logger.info("스케줄 작업을 시작합니다.")
    # schedule_jobs()
    job()
