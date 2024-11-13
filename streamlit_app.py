import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# 데이터베이스 연결 함수
def get_connection():
    return sqlite3.connect('bitcoin_trades.db')

# 데이터 로드 함수
def load_data():
    conn = get_connection()
    query = "SELECT * FROM trades ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('America/New_York')  # 뉴욕 시간으로 변환
    conn.close()
    return df

# 메인 함수
def main():
    st.title('Bitcoin Trades Viewer')

    # 원금 설정
    initial_investment = 7000000  # 700만 원

    # 데이터 로드
    df = load_data()

    # 포트폴리오 총 자산 계산
    df['total_asset'] = df['krw_balance'] + df['btc_balance'] * df['btc_krw_price']
    
    # 최신 포트폴리오 총 자산 가져오기
    latest_total_asset = df['total_asset'].iloc[0] if not df.empty else initial_investment

    # 수익률 계산
    profit_rate = ((latest_total_asset - initial_investment) / initial_investment) * 100

    # 수익률 표시
    st.subheader("포트폴리오 투자 수익률")
    st.write(f"원금: {initial_investment:,} 원")
    st.write(f"현재 총 자산: {latest_total_asset:,.0f} 원")
    st.write(f"투자 수익률: {profit_rate:.2f}%")

    # 기본 통계
    st.header('기본 통계')
    st.write(f"Total number of trades: {len(df)}")
    st.write(f"First trade date: {df['timestamp'].min()}")
    st.write(f"Last trade date: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('거래 내역')
    # 거래 내역에서 reason과 reflection 컬럼을 제거합니다
    df_display = df.drop(columns=['reason', 'reflection'])
    st.dataframe(df_display)

    # 거래 결정 분포
    st.header('거래 결정 분포')
    decision_counts = df['decision'].value_counts()
    fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='Trade Decisions')
    st.plotly_chart(fig)

   # BTC 가격 변화
    st.header('BTC 가격 변화')
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC Price (KRW)')
    st.plotly_chart(fig)

    # BTC 잔액 변화
    st.header('BTC 잔액 변화')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC Balance')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW 잔액 변화')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW Balance')
    st.plotly_chart(fig)

    # 포트폴리오 총 자산 변화
    st.header('포트폴리오 총 자산 변화')
    df['total_asset'] = df['krw_balance'] + df['btc_balance'] * df['btc_krw_price']
    fig = px.line(df, x='timestamp', y='total_asset', title='총 자산 가치 (KRW)')
    st.plotly_chart(fig)

    # 거래 이유 및 반성 내용 표시
    st.header('거래 이유 및 반성 내용')
    selected_trade = st.selectbox('거래 선택', df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    trade_details = df[df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') == selected_trade]
    if not trade_details.empty:
        st.subheader('거래 이유')
        st.write(trade_details.iloc[0]['reason'])
        st.subheader('반성 내용')
        st.write(trade_details.iloc[0]['reflection'])

if __name__ == "__main__":
    main()
