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
    query = "SELECT * FROM trades"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# 메인 함수
def main():
    st.title('Bitcoin Trades Viewer')

    # 데이터 로드
    df = load_data()

    # 날짜 컬럼을 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 기본 통계
    st.header('기본 통계')
    st.write(f"총 거래 횟수: {len(df)}")
    st.write(f"첫 거래 날짜: {df['timestamp'].min()}")
    st.write(f"마지막 거래 날짜: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('거래 내역')
    # 필요한 컬럼 선택 (새로운 컬럼 포함)
    trade_columns = [
        'timestamp', 'decision', 'target_btc_ratio', 'current_btc_ratio',
        'difference', 'executed_percentage', 'reason', 'btc_balance',
        'krw_balance', 'btc_avg_buy_price', 'btc_krw_price', 'reflection'
    ]
    st.dataframe(df[trade_columns])

    # 거래 결정 분포
    st.header('거래 결정 분포')
    decision_counts = df['decision'].value_counts()
    fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='거래 결정 비율')
    st.plotly_chart(fig)

    # 목표 비중 vs 현재 비중 변화
    st.header('목표 비중과 현재 비중의 변화')
    fig = px.line(df, x='timestamp', y=['target_btc_ratio', 'current_btc_ratio'], title='목표 vs 현재 비트코인 비중')
    st.plotly_chart(fig)

    # 비중 차이(difference) 변화
    st.header('비중 차이 (Difference) 변화')
    fig = px.line(df, x='timestamp', y='difference', title='목표와 현재 비중의 차이')
    st.plotly_chart(fig)

    # 거래 실행 비중 분포
    st.header('실제 거래된 비중 (Executed Percentage) 분포')
    fig = px.histogram(df, x='executed_percentage', nbins=20, title='Executed Percentage Distribution')
    st.plotly_chart(fig)

    # BTC 잔액 변화
    st.header('BTC 잔액 변화')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC 잔액')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW 잔액 변화')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW 잔액')
    st.plotly_chart(fig)

    # 포트폴리오 총 자산 변화
    st.header('포트폴리오 총 자산 가치 변화')
    df['total_asset'] = df['krw_balance'] + df['btc_balance'] * df['btc_krw_price']
    fig = px.line(df, x='timestamp', y='total_asset', title='총 자산 가치 (KRW)')
    st.plotly_chart(fig)

    # BTC 가격 변화
    st.header('BTC 가격 변화')
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC 가격 (KRW)')
    st.plotly_chart(fig)

    # 거래 이유 및 반성 내용 표시 (옵션)
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

# 실행 : TERMINAL에 "streamlit run ./streamlit_app.py"