import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import base64
from PIL import Image
import io
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException, NoSuchElementException
import logging
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from openai import OpenAI
import sqlite3
from datetime import datetime, timedelta
import schedule
import ccxt
import pytz

class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  btc_balance REAL,
                  krw_balance REAL,
                  btc_avg_buy_price REAL,
                  btc_krw_price REAL)''')
    conn.commit()
    return conn

def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    c = conn.cursor()
    ny_time = datetime.now(pytz.timezone('America/New_York')).isoformat()  # 모듈 이름 사용
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (ny_time, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
    conn.commit()

def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0
    
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    
    return (final_balance - initial_balance) / initial_balance * 100

def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)
    
    # 최근 거래 데이터 요약 (최신 5개만 추출하고 필요한 열만 선택)
    recent_trades_summary = trades_df[['timestamp', 'decision', 'percentage', 'reason']].tail(5).to_json(orient='records')
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
                Recent trading data summary:
                {recent_trades_summary}
                
                Current market data:
                {current_market_data}
                
                Overall performance in the last 7 days: {performance:.2f}%
                
                Please analyze this data and provide:
                1. A brief reflection on the recent trading decisions
                2. Insights on what worked well and what didn't
                3. Suggestions for improvement in future trading decisions
                4. Any patterns or trends you notice in the market data
                
                Limit your response to 250 words or less.
                """
            }
        ]
    )
    
    return response.choices[0].message.content

def get_db_connection():
    return sqlite3.connect('bitcoin_trades.db')

# 데이터베이스 초기화
init_db()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def add_indicators(df):
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
    
    return df

def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]
    else:
        logger.error(f"Failed to fetch Fear and Greed Index. Status code: {response.status_code}")
        return None

# 차트 캡쳐
def create_driver():
    logger.info("ChromeDriver 설정 중...")
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 헤드리스 모드 사용
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        service = Service('/usr/bin/chromedriver')  # Specify the path to the ChromeDriver executable

        # Initialize the WebDriver with the specified options
        driver = webdriver.Chrome(service=service, options=chrome_options)

         # webdriver-manager를 사용하여 ChromeDriver 자동 설치 및 버전 관리
        service = Service(ChromeDriverManager().install())

        # WebDriver 초기화
        driver = webdriver.Chrome(service=service, options=chrome_options)

        return driver
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

def capture_and_encode_screenshot(driver, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 스크린샷 캡처
            png = driver.get_screenshot_as_png()
            
            # PIL Image로 변환
            img = Image.open(io.BytesIO(png))
            
            # 이미지 포맷 확인
            if img.format != 'PNG':
                logger.error("이미지 형식이 PNG가 아님. 지원되는 형식으로 변환 필요.")
                img = img.convert("RGB")

            # 이미지 리사이즈 (OpenAI API 제한에 맞춤)
            img.thumbnail((2000, 2000))
            
            # 이미지를 바이트로 변환
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            
            # base64로 인코딩
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return base64_image
        except Exception as e:
            logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
            retry_count += 1
            time.sleep(2)  # 잠시 대기 후 재시도

    logger.error("최대 재시도 횟수에 도달하여 스크린샷 캡처 실패")
    return None

def ai_trading():
    # Upbit 객체 생성
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)

    # 1. 현재 투자 상태 조회
    # all_balances = upbit.get_balances()
    # filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
    
    all_balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in all_balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in all_balances if balance['currency'] == 'KRW'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")

    # 전체 자산 및 비중 계산
    btc_value_krw = btc_balance * current_btc_price  # 비트코인 자산 가치
    total_asset_value = krw_balance + btc_value_krw  # 총 자산 가치
    btc_ratio = (btc_value_krw / total_asset_value) * 100 if total_asset_value > 0 else 0  # BTC 비중(%)
    krw_ratio = (krw_balance / total_asset_value) * 100 if total_asset_value > 0 else 0  # KRW 비중(%)

     # 보유 자산 비중을 출력
    logger.info(f"BTC 비중: {btc_ratio:.2f}%, KRW 비중: {krw_ratio:.2f}%")

    # 2. 오더북(호가 데이터) 조회
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    
    # 3. 차트 데이터 조회 및 보조지표 추가
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)
    
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)

    # 4. ccxt로 USD-BTC 데이터 가져오기 (Kraken)
    kraken = ccxt.kraken()
    ohlcv_usd = kraken.fetch_ohlcv("BTC/USD", timeframe='1d', limit=30)
    
    # USD 일봉 데이터
    ohlcv_usd_daily = kraken.fetch_ohlcv("BTC/USD", timeframe='1d', limit=30)
    df_usd_daily = pd.DataFrame(ohlcv_usd_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_usd_daily['timestamp'] = pd.to_datetime(df_usd_daily['timestamp'], unit='ms')
    df_usd_daily.set_index('timestamp', inplace=True)
    df_usd_daily = dropna(df_usd_daily)
    df_usd_daily = add_indicators(df_usd_daily)  # USD 일봉 데이터에 지표 추가

    # USD 시간봉 데이터
    ohlcv_usd_hourly = kraken.fetch_ohlcv("BTC/USD", timeframe='1h', limit=24)
    df_usd_hourly = pd.DataFrame(ohlcv_usd_hourly, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_usd_hourly['timestamp'] = pd.to_datetime(df_usd_hourly['timestamp'], unit='ms')
    df_usd_hourly.set_index('timestamp', inplace=True)
    df_usd_hourly = dropna(df_usd_hourly)
    df_usd_hourly = add_indicators(df_usd_hourly)  # USD 시간봉 데이터에 지표 추가

    # 5. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    # 7. YouTube 자막 데이터 가져오기
    f = open("strategy.txt", "r", encoding="utf-8") # 직접 저장한 텍스트를 넣어주기
    youtube_transcript = f.read()
    f.close()

    # Selenium으로 차트 캡처
    driver = None
    usd_chart_image = None
    krw_chart_image = None
    try:
        driver = create_driver()
        
        # USD 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.USDT-BTC")
        logger.info("USD 차트 페이지 로드 완료")
        time.sleep(5)  # 페이지 로딩 대기 시간 증가
        logger.info("USD 차트 캡처 시작")
        usd_chart_image = capture_and_encode_screenshot(driver)
        logger.info("USD 차트 캡처 완료")

        # KRW 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC")
        logger.info("KRW 차트 페이지 로드 완료")
        time.sleep(5)  # 페이지 로딩 대기 시간 증가
        logger.info("KRW 차트 캡처 시작")
        krw_chart_image = capture_and_encode_screenshot(driver)
        logger.info("KRW 차트 캡처 완료")

    except WebDriverException as e:
        logger.error(f"WebDriver 오류 발생: {e}")
    except Exception as e:
        logger.error(f"차트 캡처 중 오류 발생: {e}")
    finally:
        if driver:
            driver.quit()

    # AI에게 데이터 제공하고 판단 받기
    client = OpenAI()

    # 데이터베이스 연결
    conn = get_db_connection()
    
    # 최근 거래 내역 가져오기
    recent_trades = get_recent_trades(conn)
    
    # 현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
    current_market_data = {
        "fear_greed_index": fear_greed_index,
        "orderbook": orderbook,
        "btc_ratio": btc_ratio,   # BTC 비중
        "krw_ratio": krw_ratio,   # KRW 비중
        "daily_ohlcv": df_daily.to_dict(),
        "hourly_ohlcv": df_hourly.to_dict(),
        "daily_ohlcv_usd": df_usd_daily.to_dict(),  # USD 일봉 데이터
        "hourly_ohlcv_usd": df_usd_hourly.to_dict()  # USD 시간봉 데이터
    }
    
    # 반성 및 개선 내용 생성
    reflection = generate_reflection(recent_trades, current_market_data)
    
    # AI 모델에 반성 내용 제공
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert in Bitcoin investing. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:
                - Technical indicators and market data (both KRW-BTC and USD-BTC)
                - The Fear and Greed Index and its implications
                - Overall market sentiment
                - Patterns and trends visible in the chart image
                - BTC and KRW balance ratios in the current portfolio
                - Recent trading performance and reflection

                Recent trading reflection:
                {reflection}

                Particularly important is to always refer to the trading method of 'Wonyyotti', a legendary Korean investor, to assess the current situation and make trading decisions. Wonyyotti's trading method is as follows:

                {youtube_transcript}

                Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data and recent performance reflection.

                Response format:
                1. Decision (buy, sell, or hold)
                2. If the decision is 'buy', provide a percentage (1-100) of available KRW to use for buying.
                If the decision is 'sell', provide a percentage (1-100) of held BTC to sell.
                If the decision is 'hold', set the percentage to 0.
                3. Reason for your decision

                Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                Your percentage should reflect the strength of your conviction in the decision based on the analyzed data."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Current investment status: BTC balance: {btc_balance} BTC, KRW balance: {krw_balance} KRW
        Orderbook: {json.dumps(orderbook)}
        BTC ratio: {btc_ratio}%
        KRW ratio: {krw_ratio}%
        Daily OHLCV with indicators (30 days): {df_daily.to_json()}
        Hourly OHLCV with indicators (24 hours): {df_hourly.to_json()}
        Daily OHLCV with indicators (USD-BTC): {df_usd_daily.to_json()}
        Hourly OHLCV with indicators (USD-BTC): {df_usd_hourly.to_json()}
        Fear and Greed Index: {json.dumps(fear_greed_index)}"""
                    },
                    {
                        "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{usd_chart_image}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{krw_chart_image}"
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "trading_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "percentage": {"type": "integer"},
                        "reason": {"type": "string"}
                    },
                    "required": ["decision", "percentage", "reason"],
                    "additionalProperties": False
                }
            }
        },
        max_tokens=4095
    )

    # 최신 pydantic 메서드 사용
    result = TradingDecision.model_validate_json(response.choices[0].message.content)

    print(f"### AI Decision: {result.decision.upper()} ###")
    print(f"### Reason: {result.reason} ###")

    order_executed = False

    if result.decision == "buy":
        my_krw = upbit.get_balance("KRW")
        buy_amount = my_krw * (result.percentage / 100) * 0.9995  # 수수료 고려
        if buy_amount > 5000:
            print(f"### Buy Order Executed: {result.percentage}% of available KRW ###")
            order = upbit.buy_market_order("KRW-BTC", buy_amount)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif result.decision == "sell":
        my_btc = upbit.get_balance("KRW-BTC")
        sell_amount = my_btc * (result.percentage / 100)
        current_price = pyupbit.get_current_price("KRW-BTC")
        if sell_amount * current_price > 5000:
            print(f"### Sell Order Executed: {result.percentage}% of held BTC ###")
            order = upbit.sell_market_order("KRW-BTC", sell_amount)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW worth) ###")

    # 거래 실행 여부와 관계없이 현재 잔고 조회
    time.sleep(1)  # API 호출 제한을 고려하여 잠시 대기
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")

    # 거래 정보 및 반성 내용 로깅
    log_trade(conn, result.decision, result.percentage if order_executed else 0, result.reason, 
              btc_balance, krw_balance, btc_avg_buy_price, current_btc_price, reflection)

    # 데이터베이스 연결 종료
    conn.close()

def job():
    try:
        ai_trading()
    except Exception as e:
        logger.error(f"An error occurredL {e}")

# 테스트용 바로 실행
job()

# # 매일 특정 시간(예: 오전 9시, 오후 3시, 오후 9시)에 실행
# schedule.every().day.at("00:00").do(job)
# schedule.every().day.at("04:00").do(job)
# schedule.every().day.at("08:00").do(job)
# schedule.every().day.at("12:00").do(job)
# schedule.every().day.at("16:00").do(job)
# schedule.every().day.at("20:00").do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)
