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
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException, NoSuchElementException
import logging
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
import sqlite3
import schedule
import yfinance as yf

class TradingDecision(BaseModel):
    target_btc_ratio: float
    reason: str

def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  current_btc_ratio REAL,
                  target_btc_ratio REAL,
                  btc_ratio_difference REAL,
                  reason TEXT,
                  btc_balance REAL,
                  krw_balance REAL,
                  btc_avg_buy_price REAL,
                  btc_krw_price REAL,
                  reflection TEXT)''')
    conn.commit()
    return conn

def log_trade(conn, current_btc_ratio, target_btc_ratio, btc_ratio_difference, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, current_btc_ratio, target_btc_ratio, btc_ratio_difference, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, current_btc_ratio, target_btc_ratio, btc_ratio_difference, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
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

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
                Recent trading data:
                {trades_df.to_json(orient='records')}
                
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

    # MFI (Money Flow Index)
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
    
    # 이동평균선 (EMA 20일, 50일, 120일, 200일)
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

def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")

def perform_chart_actions(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    
    # 1시간 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
        "1시간 옵션"
    )
    
    # 지표 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    
    # 볼린저 밴드 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )

def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        
        # 이미지 리사이즈 (OpenAI API 제한에 맞춤)
        img.thumbnail((2000, 2000))
        
        # 현재 시간을 파일명에 포함
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upbit_chart_{current_time}.png"
        
        # 현재 스크립트의 경로를 가져옴
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 이미지를 바이트로 변환
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        
        # base64로 인코딩
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
        return None

def get_combined_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        combined_text = ' '.join(entry['text'] for entry in transcript)
        return combined_text
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {e}")
        return ""

def ai_trading():
    # Upbit 객체 생성
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)

    # 1. 현재 투자 상태 조회
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
    
    # 2. 오더북(호가 데이터) 조회
    orderbook = pyupbit.get_orderbook("KRW-BTC")

    # 3. 환율 정보 가져오기 (USD/KRW)
    usdkrw = yf.Ticker("KRW=X")
    usdkrw_rate = usdkrw.history(period="1d")['Close'][-1]
    logger.info(f"현재 USD/KRW 환율: {usdkrw_rate}")

    # 4. KRW 비트코인 가격 가져오기
    krw_btc_price = pyupbit.get_current_price("KRW-BTC")
    logger.info(f"현재 KRW-BTC 가격: {krw_btc_price}")

    # 5. USD 비트코인 가격 가져오기
    btc_usd_price = yf.Ticker("BTC-USD").history(period="1d")['Close'][-1]
    logger.info(f"현재 BTC-USD 가격: {btc_usd_price}")

    # 6. USD 비트코인 가격을 KRW로 환산
    usd_btc_price_in_krw = btc_usd_price * usdkrw_rate
    logger.info(f"USD 기준 BTC 가격의 KRW 환산값: {usd_btc_price_in_krw}")

    # 7. 가격 차이 계산 (프리미엄 또는 할인율)
    price_difference = ((krw_btc_price - usd_btc_price_in_krw) / usd_btc_price_in_krw) * 100
    logger.info(f"KRW-BTC와 USD-BTC 가격 차이율: {price_difference:.2f}%")

    # 8. 차트 데이터 조회 및 보조지표 추가 (USD 기준)
    btc_usd = yf.Ticker("BTC-USD")
    df_daily_usd = btc_usd.history(period="200d", interval="1d")
    df_daily_usd.reset_index(inplace=True)
    df_daily_usd = df_daily_usd.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df_daily_usd = dropna(df_daily_usd)
    df_daily_usd = add_indicators(df_daily_usd)

    # 9. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    # 10. 뉴스 헤드라인 가져오기
    news_headlines = get_bitcoin_news()

    # 11. YouTube 자막 데이터 가져오기
    with open("strategy.txt", "r", encoding="utf-8") as f:
        youtube_transcript = f.read()

    # 12. Selenium으로 차트 캡처
    driver = None
    try:
        driver = create_driver()
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC")
        logger.info("페이지 로드 완료")
        time.sleep(5)  # 페이지 로딩 대기 시간 증가
        logger.info("차트 작업 시작")
        perform_chart_actions(driver)
        logger.info("차트 작업 완료")
        chart_image = capture_and_encode_screenshot(driver)
        logger.info(f"스크린샷 캡처 완료.")
    except WebDriverException as e:
        logger.error(f"WebDriver 오류 발생: {e}")
        chart_image = None
    except Exception as e:
        logger.error(f"차트 캡처 중 오류 발생: {e}")
        chart_image = None
    finally:
        if driver:
            driver.quit()

    # AI에게 데이터 제공하고 판단 받기
    client = OpenAI()

    # 데이터베이스 연결
    conn = get_db_connection()
    
    # 최근 거래 내역 가져오기
    recent_trades = get_recent_trades(conn)
    
    # 현재 시장 데이터 수집
    current_market_data = {
        "fear_greed_index": fear_greed_index,
        "news_headlines": news_headlines,
        "orderbook": orderbook,
        "price_difference": price_difference,
        "krw_btc_price": krw_btc_price,
        "usd_btc_price_in_krw": usd_btc_price_in_krw,
        "daily_ohlcv_usd": df_daily_usd.tail(5).to_dict(),
    }
    
    # 반성 및 개선 내용 생성
    reflection = generate_reflection(recent_trades, current_market_data)
    
    # AI 모델에 반성 내용 제공
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert in Bitcoin investing. Analyze the provided data and determine the ideal proportion of Bitcoin to hold in the portfolio at the current moment. Consider the following in your analysis:

                - Technical indicators and market data (USD)
                - The price difference between KRW and USD Bitcoin prices
                - Recent news headlines and their potential impact on Bitcoin price
                - The Fear and Greed Index and its implications
                - Overall market sentiment
                - Patterns and trends visible in the chart image
                - Recent trading performance and reflection

                Recent trading reflection:
                {reflection}

                Particularly important is to always refer to the trading method of 'Wonyyotti', a legendary Korean investor, to assess the current situation and make trading decisions. Wonyyotti's trading method is as follows:

                {youtube_transcript}

                Based on this trading method, analyze the current market situation and determine the ideal proportion of Bitcoin to hold in the portfolio (as a decimal between 0.0 and 1.0).

                Response format:
                1. Target Bitcoin Ratio (decimal between 0.0 and 1.0)
                2. Reason for your decision

                Ensure that the target ratio is a decimal number between 0.0 and 1.0.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Current investment status: {json.dumps(filtered_balances)}
Orderbook: {json.dumps(orderbook)}
KRW-BTC Price: {krw_btc_price}
USD-BTC Price in KRW: {usd_btc_price_in_krw}
Price Difference (%): {price_difference:.2f}
Daily OHLCV with indicators (USD): {df_daily_usd.tail(5).to_json()}
Recent news headlines: {json.dumps(news_headlines)}
Fear and Greed Index: {json.dumps(fear_greed_index)}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{chart_image}"
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
                        "target_btc_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"}
                    },
                    "required": ["target_btc_ratio", "reason"],
                    "additionalProperties": False
                }
            }
        },
        max_tokens=1000
    )

    # 최신 pydantic 메서드 사용
    result = TradingDecision.model_validate_json(response.choices[0].message.content)

    print(f"### Target BTC Ratio: {result.target_btc_ratio} ###")
    print(f"### Reason: {result.reason} ###")

    # 현재 잔고 조회
    btc_balance = float(upbit.get_balance("KRW-BTC"))
    krw_balance = float(upbit.get_balance("KRW"))

    # 현재 비트코인 가격
    current_price = pyupbit.get_current_price("KRW-BTC")

    # 총자산 계산
    total_asset = krw_balance + btc_balance * current_price

    # 현재 비트코인 비중 계산
    if total_asset > 0:
        current_btc_ratio = (btc_balance * current_price) / total_asset
    else:
        current_btc_ratio = 0

    # 목표 비트코인 비중
    target_btc_ratio = result.target_btc_ratio

    # 필요한 비트코인 양 계산
    target_btc_balance = (total_asset * target_btc_ratio) / current_price
    btc_to_buy_or_sell = target_btc_balance - btc_balance

    # 비트코인 비중 차이 계산
    btc_ratio_difference = abs(target_btc_ratio - current_btc_ratio)

    order_executed = False

    if btc_ratio_difference > 0.1:
        if abs(btc_to_buy_or_sell * current_price) < 5000:
            print("### Trade Skipped: Amount less than 5000 KRW ###")
        else:
            if btc_to_buy_or_sell > 0:
                # 매수 주문
                buy_amount = btc_to_buy_or_sell * current_price * 1.0005  # 수수료 고려
                if krw_balance >= buy_amount:
                    print(f"### Buy Order Executed: Buying {btc_to_buy_or_sell} BTC ###")
                    order = upbit.buy_market_order("KRW-BTC", buy_amount)
                    if order:
                        order_executed = True
                    print(order)
                else:
                    print("### Buy Order Failed: Insufficient KRW ###")
            elif btc_to_buy_or_sell < 0:
                # 매도 주문
                sell_amount = abs(btc_to_buy_or_sell)
                if btc_balance >= sell_amount:
                    print(f"### Sell Order Executed: Selling {sell_amount} BTC ###")
                    order = upbit.sell_market_order("KRW-BTC", sell_amount)
                    if order:
                        order_executed = True
                    print(order)
                else:
                    print("### Sell Order Failed: Insufficient BTC ###")
    else:
        print("### No Trade Needed: Difference less than 10% threshold ###")

    # 거래 실행 여부와 관계없이 현재 잔고 조회
    time.sleep(1)  # API 호출 제한을 고려하여 잠시 대기
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")

    # 거래 정보 및 반성 내용 로깅
    log_trade(conn, current_btc_ratio, target_btc_ratio, btc_ratio_difference, result.reason, 
    btc_balance, krw_balance, btc_avg_buy_price, current_btc_price, reflection)

    # 데이터베이스 연결 종료
    conn.close()

def job():
    try:
        ai_trading()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

# 테스트용 바로 실행
job()

# # 매일 특정 시간에 실행
# schedule.every().day.at("00:00").do(job)
# schedule.every().day.at("04:00").do(job)
# schedule.every().day.at("08:00").do(job)
# schedule.every().day.at("12:00").do(job)
# schedule.every().day.at("16:00").do(job)
# schedule.every().day.at("20:00").do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)