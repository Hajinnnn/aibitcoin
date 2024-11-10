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
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
import sqlite3
import schedule
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import ccxt
import pytz
import asyncio
import websockets
import threading

# 환경 변수 로드
load_dotenv()

# Upbit 객체를 전역에서 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(access, secret)

class PortfolioAllocation(BaseModel):
    target_btc_ratio: float
    reason: str

def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
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
    conn.close()

def log_trade(decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, 
              target_btc_ratio=None, current_btc_ratio=None, difference=None, executed_percentage=None, reflection=''):
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        c = conn.cursor()
        
        # 뉴욕 시간대로 타임스탬프 생성
        ny_timezone = pytz.timezone("America/New_York")
        timestamp = datetime.now(ny_timezone).isoformat()

        # 중복 검사를 제거하고 모든 기록을 저장
        c.execute("""
            INSERT INTO trades 
            (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, 
             target_btc_ratio, current_btc_ratio, difference, executed_percentage, reflection) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, 
              target_btc_ratio, current_btc_ratio, difference, executed_percentage, reflection))
        conn.commit()
    except Exception as e:
        print(f"Error logging trade: {e}")
    finally:
        conn.close()

def get_recent_trades(days=7):
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    df = pd.DataFrame.from_records(data=c.fetchall(), columns=columns)
    conn.close()

     # timestamp 컬럼을 UTC 시간으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    return df

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0

    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']

    return (final_balance - initial_balance) / initial_balance * 100

def generate_reflection(trades_df, current_market_data, wonyyotti_strategy):
    performance = calculate_performance(trades_df)

    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an AI trading assistant tasked with analyzing recent trading performance and "
                "current market conditions to generate insights and improvements for future trading decisions."
            )
        },
        {
            "role": "user",
            "content": f"""
                Recent trading data:
                {trades_df.to_json(orient='records')}

                Current market data:
                {current_market_data}

                Overall performance in the last 7 days: {performance:.2f}%

                Trading strategy reference (from 워뇨띠's YouTube trading strategy):
                {wonyyotti_strategy}

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
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Fear and Greed Index. Error: {e}")
        return None

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
        headlines = []
        for item in news_results:
            headlines.append({
                "title": item.get("title", ""),
                "date": item.get("date", "")
            })

        return headlines[:5]
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []

# #로컬용
# def setup_chrome_options():
#     chrome_options = Options()
#     chrome_options.add_argument("--start-maximized")
#     chrome_options.add_argument("--headless")  # 디버깅 시 주석 처리 가능
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
#     return chrome_options

# def create_driver():
#     logger.info("ChromeDriver 설정 중...")
#     service = Service(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=setup_chrome_options())
#     return driver

# EC2 서버용
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

        return driver
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

# def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
#     try:
#         element = WebDriverWait(driver, wait_time).until(
#             EC.presence_of_element_located((By.XPATH, xpath))
#         )
#         driver.execute_script("arguments[0].scrollIntoView(true);", element)
#         element = WebDriverWait(driver, wait_time).until(
#             EC.element_to_be_clickable((By.XPATH, xpath))
#         )
#         element.click()
#         logger.info(f"{element_name} 클릭 완료")
#         time.sleep(2)
#     except TimeoutException:
#         logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
#     except ElementClickInterceptedException:
#         logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
#     except NoSuchElementException:
#         logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
#     except Exception as e:
#         logger.error(f"{element_name} 클릭 중 오류 발생: {e}")

# def perform_chart_actions(driver):
#     # 시간 메뉴 클릭
#     click_element_by_xpath(
#         driver,
#         "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
#         "시간 메뉴"
#     )

#     # 1시간 옵션 선택
#     click_element_by_xpath(
#         driver,
#         "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
#         "1시간 옵션"
#     )

#     # 지표 메뉴 클릭
#     click_element_by_xpath(
#         driver,
#         "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
#         "지표 메뉴"
#     )

#     # 볼린저 밴드 옵션 선택
#     click_element_by_xpath(
#         driver,
#         "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
#         "볼린저 밴드 옵션"
#     )

# def capture_and_encode_screenshot():
#     try:
#         driver = create_driver()
#         driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC")
#         logger.info("페이지 로드 완료")
#         time.sleep(5)
#         logger.info("차트 작업 시작")
#         perform_chart_actions(driver)
#         logger.info("차트 작업 완료")

#         png = driver.get_screenshot_as_png()
#         img = Image.open(io.BytesIO(png))
#         img.thumbnail((2000, 2000))
#         buffered = io.BytesIO()
#         img.save(buffered, format="PNG")
#         base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

#         logger.info(f"스크린샷 캡처 완료.")
#         driver.quit()
#         return base64_image
#     except WebDriverException as e:
#         logger.error(f"WebDriver 오류 발생: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"차트 캡처 중 오류 발생: {e}")
#         return None

# #로컬용
# def get_combined_transcript(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
#         combined_text = ' '.join(entry['text'] for entry in transcript)
#         return combined_text
#     except Exception as e:
#         logger.error(f"Error fetching YouTube transcript: {e}")
#         return ""

#EC2용
def get_combined_transcript():
    # EC2 서버용: 로컬 텍스트 파일에서 스크립트 읽어오기
    try:
        with open("strategy.txt", "r", encoding="utf-8") as f:
            combined_text = f.read()
        return combined_text
    except Exception as e:
        logger.error(f"Error reading transcript from file: {e}")
        return ""

def send_alert(message):
    # 알림을 발송하는 함수 (예: 텔레그램)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    params = {
        'chat_id': telegram_chat_id,
        'text': message
    }
    try:
        requests.get(url, params=params)
        logger.info("알림 발송 성공")
    except Exception as e:
        logger.error(f"알림 발송 실패: {e}")

def calculate_support_resistance(df):
    # 저항선과 지지선을 계산 (USD 차트를 기준으로)
    n = 20  # 최근 20개의 봉을 사용
    recent_high = df['high'].rolling(window=n).max()
    recent_low = df['low'].rolling(window=n).min()
    return recent_high.iloc[-1], recent_low.iloc[-1]

def get_krw_balance():
    # KRW 잔액 조회
    balances = upbit.get_balances()
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    return krw_balance

def get_btc_balance():
    # BTC 잔액 조회
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    return btc_balance

def execute_buy_order(amount):
    # 손절매 및 이익 실현 가격 설정
    current_price = pyupbit.get_current_price("KRW-BTC")
    stop_loss_price = current_price * 0.90  # 10% 손절
    take_profit_price = current_price * 1.10  # 10% 이익 실현

    if amount > 5000:
        order = upbit.buy_market_order("KRW-BTC", amount)
        logger.info(f"매수 주문 실행: {amount} KRW")
        # 손절매 및 이익 실현 주문 설정 (업비트는 지원하지 않으므로 수동 구현 필요)
        # 매수 후 손절매 및 이익 실현 모니터링 시작
        monitor_position(stop_loss_price, take_profit_price, amount)
        return order
    else:
        logger.info("매수 금액이 최소 주문 금액보다 적습니다.")
        return None

def execute_sell_order(amount):
    # 분할 매도를 실행하는 함수
    current_price = pyupbit.get_current_price("KRW-BTC")
    if amount * current_price > 5000:
        order = upbit.sell_market_order("KRW-BTC", amount)
        logger.info(f"매도 주문 실행: {amount} BTC")
        send_alert(f"매도 주문 실행: {amount} BTC")
        return order
    else:
        logger.info("매도 금액이 최소 주문 금액보다 적습니다.")
        return None

def calculate_trade_amount(balance, risk_percentage):
    # 잔고의 일정 비율을 매매 금액으로 사용
    return balance * (risk_percentage / 100)

def monitor_position(stop_loss_price, take_profit_price, amount):
    while True:
        # 현재 가격을 실시간으로 가져오기
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        # 손절매 조건: 현재 가격이 손절 가격 이하로 떨어질 경우
        if current_price <= stop_loss_price:
            order = upbit.sell_market_order("KRW-BTC", amount)
            logger.info(f"손절매 주문 실행: {amount} BTC @ {current_price} KRW")
            break  # 손절매 후 모니터링 종료
        
        # 이익 실현 조건: 현재 가격이 목표 가격 이상으로 상승할 경우
        elif current_price >= take_profit_price:
            order = upbit.sell_market_order("KRW-BTC", amount)
            logger.info(f"이익 실현 주문 실행: {amount} BTC @ {current_price} KRW")
            break  # 이익 실현 후 모니터링 종료
        
        # 1초 대기 후 다시 확인 (가격을 실시간으로 모니터링)
        time.sleep(1)

def update_resistance_support(current_price, resistance, support, breakout_margin=0.01):
    if current_price > resistance * (1 + breakout_margin):
        # 강한 상승 돌파, 저항선 업데이트
        resistance = current_price
    elif current_price < support * (1 - breakout_margin):
        # 강한 하락 돌파, 지지선 업데이트
        support = current_price
    return resistance, support


def is_buy_signal(df):
    # 기술 지표 기반 매수 신호 감지
    current_rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    return current_rsi < 30 and macd > macd_signal

def is_sell_signal(df):
    # 기술 지표 기반 매도 신호 감지
    current_rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    return current_rsi > 70 and macd < macd_signal

async def real_time_price_monitoring(resistance, support, stop_event):
    # 실시간 가격 모니터링하여 저항선 돌파 및 지지선 붕괴 시 매매 실행
    uri = "wss://api.upbit.com/websocket/v1"
    async with websockets.connect(uri) as websocket:
        subscribe_data = [
            {"ticket": "test"},
            {"type": "ticker", "codes": ["KRW-BTC"]},
            {"format": "SIMPLE"}
        ]
        await websocket.send(json.dumps(subscribe_data))
        
        # 최근 가격 데이터를 저장하기 위한 리스트
        price_history = []
        max_history_length = 100  # 최대 저장할 가격 데이터 수

        while not stop_event.is_set():
            data = await websocket.recv()
            data = json.loads(data)
            current_price = data['tp']
            print(f"현재 가격: {current_price}")

            # 가격 히스토리 업데이트
            price_history.append(current_price)
            if len(price_history) > max_history_length:
                price_history.pop(0)
            
            # 지지선/저항선 업데이트
            resistance, support = update_resistance_support(current_price, resistance, support)
            print(f"현재 저항선: {resistance}, 현재 지지선: {support}")

            # 매매 신호 확인 (기술 지표 기반)
            if len(price_history) >= 20:  # 볼린저 밴드, MACD 등의 지표 계산을 위해 최소 20개 이상의 데이터 필요
                df = pd.DataFrame({'close': price_history})
                df = add_indicators(df)
                
                # 지표 상태 출력
                print(f"현재 RSI: {df['rsi'].iloc[-1]}")
                print(f"현재 MACD: {df['macd'].iloc[-1]}, Signal: {df['macd_signal'].iloc[-1]}")
                print(f"볼린저 밴드 상단: {df['bb_bbh'].iloc[-1]}, 중간: {df['bb_bbm'].iloc[-1]}, 하단: {df['bb_bbl'].iloc[-1]}")

                # 매매 신호에 따른 매수/매도
                if is_buy_signal(df):
                    krw_balance = get_krw_balance()
                    trade_amount = calculate_trade_amount(krw_balance, risk_percentage=5)
                    execute_buy_order(trade_amount)
                    print("매수 신호에 따라 매수를 실행합니다.")
                elif is_sell_signal(df):
                    btc_balance = get_btc_balance()
                    trade_amount = calculate_trade_amount(btc_balance * current_price, risk_percentage=5)
                    execute_sell_order(trade_amount / current_price)
                    print("매도 신호에 따라 매도를 실행합니다.")
            
            # 주기적으로 상태 확인
            await asyncio.sleep(1)  # 1초마다 체크 

def start_real_time_monitoring(resistance, support, stop_event):
    asyncio.run(real_time_price_monitoring(resistance, support, stop_event))

def ai_trading():

    # Wonnyotti의 전략 가져오기
    wonyyotti_strategy = get_combined_transcript()  # strategy.txt에서 내용 가져오기

    # 데이터 수집 함수 정의
    def fetch_balances():
        all_balances = upbit.get_balances()
        filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
        return filtered_balances

    def fetch_orderbook():
        orderbook = pyupbit.get_orderbook("KRW-BTC")
        return orderbook

    def fetch_daily_ohlcv():
        df_daily_krw = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
        df_daily_krw = dropna(df_daily_krw)
        df_daily_krw = add_indicators(df_daily_krw)
        return df_daily_krw

    def fetch_hourly_ohlcv():
        df_hourly_krw = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        df_hourly_krw = dropna(df_hourly_krw)
        df_hourly_krw = add_indicators(df_hourly_krw)
        return df_hourly_krw

    def fetch_daily_usd_ohlcv():
        try:
            exchange = ccxt.kraken()
            data = exchange.fetch_ohlcv('BTC/USD', timeframe='1d', limit=30)
            df_daily_usd = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_daily_usd['timestamp'] = pd.to_datetime(df_daily_usd['timestamp'], unit='ms')
            df_daily_usd.set_index('timestamp', inplace=True)
            df_daily_usd = dropna(df_daily_usd)
            df_daily_usd = add_indicators(df_daily_usd)
            return df_daily_usd
        except Exception as e:
            logger.error(f"Error fetching USD daily OHLCV data: {e}")
            return pd.DataFrame()

    def fetch_hourly_usd_ohlcv():
        try:
            exchange = ccxt.kraken()
            data = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=24)
            df_hourly_usd = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_hourly_usd['timestamp'] = pd.to_datetime(df_hourly_usd['timestamp'], unit='ms')
            df_hourly_usd.set_index('timestamp', inplace=True)
            df_hourly_usd = dropna(df_hourly_usd)
            df_hourly_usd = add_indicators(df_hourly_usd)
            return df_hourly_usd
        except Exception as e:
            logger.error(f"Error fetching USD hourly OHLCV data: {e}")
            return pd.DataFrame()
        
    def fetch_fear_and_greed():
        return get_fear_and_greed_index()

    def fetch_news():
        return get_bitcoin_news()

    # #로컬용
    # def fetch_transcript():
    #     return get_combined_transcript("3XbtEX3jUv4")

    #EC2용
    def fetch_transcript():
        return get_combined_transcript()

    # def fetch_chart_image():
    #     return capture_and_encode_screenshot()

    def fetch_usd_krw_exchange_rate():
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            data = response.json()
            return data['rates']['KRW']
        except Exception as e:
            logger.error(f"Error fetching USD/KRW exchange rate: {e}")
            return None

    # ThreadPoolExecutor를 사용하여 병렬로 데이터 수집
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(fetch_balances): 'balances',
            executor.submit(fetch_orderbook): 'orderbook',
            executor.submit(fetch_daily_ohlcv): 'daily_ohlcv',
            executor.submit(fetch_hourly_ohlcv): 'hourly_ohlcv',
            executor.submit(fetch_fear_and_greed): 'fear_greed',
            executor.submit(fetch_news): 'news',
            executor.submit(fetch_transcript): 'transcript',
            # executor.submit(fetch_chart_image): 'chart_image',
            executor.submit(fetch_usd_krw_exchange_rate): 'usd_krw_rate',
            executor.submit(fetch_daily_usd_ohlcv): 'daily_usd_ohlcv',
            executor.submit(fetch_hourly_usd_ohlcv): 'hourly_usd_ohlcv',
        }

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                logger.info(f"{key} 데이터 수집 완료")
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
    # youtube_transcript = results.get('transcript', "")
    wonyyotti_strategy = results.get('transcript', "")
    # chart_image = results.get('chart_image', None)
    usd_krw_rate = results.get('usd_krw_rate', None)

    # 필요한 컬럼만 선택
    def select_columns(df):
        return df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal']]

    df_daily_krw = select_columns(df_daily_krw)
    df_hourly_krw = select_columns(df_hourly_krw)
    df_daily_usd = select_columns(df_daily_usd)
    df_hourly_usd = select_columns(df_hourly_usd)

     # 저항선과 지지선을 USD 차트를 기반으로 계산
    if df_hourly_usd.empty:
        logger.error("시간봉 USD 데이터가 없습니다.")
        return

    resistance, support = calculate_support_resistance(df_hourly_usd)

    # DataFrame을 JSON으로 변환
    df_daily_krw_json = df_daily_krw.to_json(orient='records')
    df_hourly_krw_json = df_hourly_krw.to_json(orient='records')
    df_daily_usd_json = df_daily_usd.to_json(orient='records')
    df_hourly_usd_json = df_hourly_usd.to_json(orient='records')

    # 괴리율 계산
    if not df_hourly_usd.empty and usd_krw_rate:
        usd_price = df_hourly_usd['close'].iloc[-1]
        krw_price = pyupbit.get_current_price("KRW-BTC")
        usd_price_in_krw = usd_price * usd_krw_rate
        premium = ((krw_price - usd_price_in_krw) / usd_price_in_krw) * 100
    else:
        premium = None
        logger.error("USD OHLCV data or USD/KRW exchange rate is unavailable.")

    # premium_formatted 변수 생성
    if premium is not None:
        premium_formatted = f"{premium:.2f}"
    else:
        premium_formatted = "N/A"

    # AI에게 데이터 제공하고 목표 비중 받기
    client = OpenAI()

    # 최근 거래 내역 가져오기
    recent_trades = get_recent_trades()

    # 현재 시장 데이터 수집
    current_market_data = {
        "fear_greed_index": fear_greed_index,
        "news_headlines": news_headlines,
        "orderbook": orderbook,
        "daily_ohlcv_krw": df_daily_krw.to_dict(),
        "hourly_ohlcv_krw": df_hourly_krw.to_dict(),
        "daily_ohlcv_usd": df_daily_usd.to_dict(),
        "hourly_ohlcv_usd": df_hourly_usd.to_dict(),
        "usd_krw_rate": usd_krw_rate,
        "krw_usd_premium": premium,
    }

    # 반성 및 개선 내용 생성
    reflection = generate_reflection(recent_trades, current_market_data, wonyyotti_strategy)

    # AI 모델에 반성 내용 제공 및 목표 비중 요청
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": f"""You are a Bitcoin investing expert. Analyze the data provided and determine the **target weight (%) that Bitcoin should have in your portfolio. In your analysis, consider the following.:

                - Technical indicators for the USD and KRW markets (RSI, MACD, etc.)
                - Exchange rate between USD and KRW
                - Percentage difference between USD price converted to KRW and current KRW price
                - Recent news headlines and their impact
                - The Fear and Greed Index and what it means
                - Overall market sentiment
                - Recent trading performance and reflection
                
                Particularly important is to always refer to the trading method of 'Wonyyotti', a legendary Korean investor, to assess the current situation and make trading decisions. Wonyyotti's trading method is as follows:
                {wonyyotti_strategy}

                Recent trading reflection:
                {reflection}           

                The target weight of Bitcoin in your portfolio should be a value between 0 and 100.

                Response format:
                1. target_btc_ratio: Bitcoin target ratio (%)
                2. reason: Reason for the decision

                Please provide your response in JSON format."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Current investment status: {json.dumps(filtered_balances)}
Orderbook: {json.dumps(orderbook)}
KRW Daily OHLCV with indicators (30 days): {df_daily_krw.to_json()}
KRW Hourly OHLCV with indicators (24 hours): {df_hourly_krw.to_json()}
USD Daily OHLCV with indicators (30 days): {df_daily_usd_json}
USD Hourly OHLCV with indicators (24 hours): {df_hourly_usd_json}
USD/KRW Exchange Rate: {usd_krw_rate}
KRW-USD Premium (%): {premium_formatted}
Recent news headlines: {json.dumps(news_headlines)}
Fear and Greed Index: {json.dumps(fear_greed_index)}"""
                    }
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/png;base64,{chart_image}"
                    #     }
                    # }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "portfolio_allocation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "target_btc_ratio": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["target_btc_ratio", "reason"],
                    "additionalProperties": False
                }
            }
        },
        max_tokens=1000
    )

    # AI 응답 검증 및 처리
    result = PortfolioAllocation.model_validate_json(response.choices[0].message.content)

    target_btc_ratio = result.target_btc_ratio
    reason = result.reason

    # target_btc_ratio가 0에서 100 사이인지 검증
    if not (0 <= target_btc_ratio <= 100):
        logger.error("Invalid target_btc_ratio value. It should be between 0 and 100.")
        # 필요에 따라 예외 처리 또는 기본 값 설정
        # 예를 들어, 범위를 벗어나면 기본값으로 설정하거나, 프로그램을 종료
        target_btc_ratio = max(0, min(target_btc_ratio, 100))

    print(f"### Target BTC Allocation: {target_btc_ratio}% ###")
    print(f"### Reason: {reason} ###")

    # 현재 잔고 조회
    time.sleep(1)
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")

    # 현재 포트폴리오 비중 계산
    total_asset = krw_balance + btc_balance * current_btc_price
    current_btc_ratio = (btc_balance * current_btc_price) / total_asset * 100 if total_asset > 0 else 0

    # 목표 비중과 현재 비중의 차이 계산
    difference = target_btc_ratio - current_btc_ratio

    order_executed = False

    # AI의 결정에 따라 매매 실행
    if difference > 0:
        # 매수 실행
        buy_amount_krw = total_asset * (difference / 100)
        if buy_amount_krw > 5000:
            print(f"### Buy Order Executed: {buy_amount_krw:.2f} KRW worth of BTC ###")
            order = upbit.buy_market_order("KRW-BTC", buy_amount_krw)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Buy Order Failed: Insufficient KRW amount ###")
    else:
        # 매도 실행
        sell_amount_btc = btc_balance * (-difference / current_btc_ratio)
        if sell_amount_btc * current_btc_price > 5000:
            print(f"### Sell Order Executed: {sell_amount_btc:.8f} BTC ###")
            order = upbit.sell_market_order("KRW-BTC", sell_amount_btc)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Sell Order Failed: Insufficient BTC amount ###")
    
    # 거래 정보 및 반성 내용 로깅
    log_trade(
        decision="buy" if order_executed and difference > 0 else "sell" if order_executed and difference < 0 else "hold",
        percentage=abs(difference) if order_executed else 0,
        reason=reason,
        btc_balance=btc_balance,
        krw_balance=krw_balance,
        btc_avg_buy_price=btc_avg_buy_price,
        btc_krw_price=current_btc_price,
        target_btc_ratio=target_btc_ratio,
        current_btc_ratio=current_btc_ratio,
        difference=difference,
        executed_percentage=abs(difference) if order_executed else 0,
        reflection=reflection
    )

    # **동적 리밸런싱 시작: 실시간 가격 모니터링**
    # 종료 이벤트 생성
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=start_real_time_monitoring, args=(resistance, support, stop_event))
    monitor_thread.start()

    # 메인 스레드에서 필요한 작업 수행...
    # 예: 일정 시간 대기 후 종료
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("프로그램이 종료됩니다.")
        stop_event.set()
        monitor_thread.join()

def job():
    try:
        ai_trading()
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # 로깅 설정
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     load_dotenv()

#     # 데이터베이스 초기화
#     init_db()

#     # # 테스트용 바로 실행
#     # job()

#     # # 매일 특정 시간에 실행
#     # schedule.every().day.at("00:00").do(job)
#     # schedule.every().day.at("04:00").do(job)
#     # schedule.every().day.at("08:00").do(job)
#     # schedule.every().day.at("12:00").do(job)
#     # schedule.every().day.at("16:00").do(job)
#     # schedule.every().day.at("20:00").do(job)

#     while True:
#         schedule.run_pending()
#         time.sleep(1)

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
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    # 데이터베이스 초기화
    init_db()

    # 스케줄 작업 실행
    logger.info("스케줄 작업을 시작합니다.")
    # schedule_jobs()

    job()
