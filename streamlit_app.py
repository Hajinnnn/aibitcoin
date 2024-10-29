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

    # timestamp를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 기본 통계
    st.header('기본 통계')
    st.write(f"총 거래 횟수: {len(df)}")
    st.write(f"첫 거래 날짜: {df['timestamp'].min()}")
    st.write(f"마지막 거래 날짜: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('거래 내역')
    st.dataframe(df)

    # 목표 비트코인 비중 분포
    st.header('목표 비트코인 비중 분포')
    target_ratio_counts = df['target_btc_ratio'].value_counts(bins=10, sort=False)
    fig = px.bar(
        x=target_ratio_counts.index.astype(str),
        y=target_ratio_counts.values,
        labels={'x': 'Target BTC Ratio Bins', 'y': 'Count'},
        title='Target Bitcoin Ratio Distribution'
    )
    st.plotly_chart(fig)

    # 비트코인 비중 변화
    st.header('비트코인 비중 변화')
    fig = px.line(df, x='timestamp', y=['current_btc_ratio', 'target_btc_ratio'], 
                  labels={'value': 'BTC Ratio', 'variable': 'Type'},
                  title='Current vs Target BTC Ratio Over Time')
    st.plotly_chart(fig)

    # 비트코인 비중 차이 히스토그램
    st.header('비트코인 비중 차이 분포')
    fig = px.histogram(df, x='btc_ratio_difference', nbins=20, title='BTC Ratio Difference Distribution')
    st.plotly_chart(fig)

    # BTC 잔액 변화
    st.header('BTC 잔액 변화')
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC Balance Over Time')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW 잔액 변화')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW Balance Over Time')
    st.plotly_chart(fig)

    # 총 자산 가치 변화 (BTC + KRW)
    st.header('총 자산 가치 변화')
    df['total_asset'] = df['krw_balance'] + df['btc_balance'] * df['btc_krw_price']
    fig = px.line(df, x='timestamp', y='total_asset', title='Total Asset Value Over Time (KRW)')
    st.plotly_chart(fig)

    # BTC 가격 변화
    st.header('BTC 가격 변화')
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC Price (KRW) Over Time')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()