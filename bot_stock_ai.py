import os
import re
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import yfinance as yf
from openai import OpenAI

# --- Renderの環境変数（Environment）から読み込み ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_ACCESS_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_stock(ticker_code):
    # 数字4桁の場合は末尾に .T を付与
    if re.match(r'^\d{4}$', ticker_code):
        ticker = f"{ticker_code}.T"
    else:
        ticker = ticker_code.upper()

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="100d")

        if len(hist) < 75:
            return f"銘柄コード【{ticker_code}】のデータが取得できないか、上場期間が短すぎます。"

        # 移動平均線
        hist['SMA5'] = hist['Close'].rolling(window=5).mean()
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA75'] = hist['Close'].rolling(window=75).mean()

        # RSI (14日)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド (20日, 2σ)
        std20 = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['SMA20'] + (std20 * 2)
        hist['BB_Lower'] = hist['SMA20'] - (std20 * 2)

        # MACD (12日, 26日, 9日)
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema12 - ema26
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]

        latest_price = latest['Close']
        prev_price = prev['Close']
        change_percent = ((latest_price - prev_price) / prev_price) * 100

        # 出来高変化率（前日比）
        vol_change = ((latest['Volume'] - prev['Volume']) / prev['Volume']) * 100 if prev['Volume'] > 0 else 0

        sma20 = latest['SMA20']
        sma75 = latest['SMA75']
        rsi = latest['RSI']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        prev_macd = prev['MACD']
        prev_signal = prev['MACD_Signal']

        dev_sma20 = ((latest_price - sma20) / sma20) * 100
        dev_sma75 = ((latest_price - sma75) / sma75) * 100

        # MACDのクロス状態判定
        macd_cross = "変化なし"
        if prev_macd < prev_signal and macd > macd_signal:
            macd_cross = "ゴールデンクロス発生（買いシグナル）"
        elif prev_macd > prev_signal and macd < macd_signal:
            macd_cross = "デッドクロス発生（売りシグナル）"
        elif macd > macd_signal:
            macd_cross = "上昇トレンド継続中（MACD > シグナル）"
        else:
            macd_cross = "下降トレンド継続中（MACD < シグナル）"

        prompt = f"""
        銘柄コード {ticker} のテクニカル分析を行ってください。

        【株価・指標データ】
        - 現在値: {latest_price:.1f}円 (前日比: {change_percent:+.2f}%)
        - 出来高変化: 前日比 {vol_change:+.1f}%
        - 20日移動平均線: {sma20:.1f}円 (乖離率: {dev_sma20:+.2f}%)
        - 75日移動平均線: {sma75:.1f}円 (乖離率: {dev_sma75:+.2f}%)
        - RSI(14日): {rsi:.1f}% (30%以下は売られすぎ、70%以上は買われすぎ)
        - ボリンジャーバンド(+2σ): {bb_upper:.1f}円 / (-2σ): {bb_lower:.1f}円
        - MACD状態: {macd_cross} (MACD: {macd:.2f}, Signal: {macd_signal:.2f})

        【指示】
        上記の指標（移動平均線、RSI、ボリンジャーバンド、MACD、出来高）を総合的に判断し、チャートの現状と今後の短期見通し・売買判断のポイントを200文字以内で論理的に記述してください。
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        ai_comment = response.choices[0].message.content.strip()

        return (
            f"■ 銘柄分析: {ticker}\n"
            f"株価: {latest_price:.1f}円 ({change_percent:+.2f}%)\n"
            f"RSI: {rsi:.1f}% | 出来高比: {vol_change:+.1f}%\n"
            f"20日線乖離: {dev_sma20:+.1f}% | 75日線乖離: {dev_sma75:+.1f}%\n"
            f"ボリンジャー: +2σ({bb_upper:.1f}円) / -2σ({bb_lower:.1f}円)\n"
            f"MACD: {macd_cross}\n\n"
            f"【AIテクニカル分析】\n{ai_comment}"
        )

    except Exception as e:
        return f"エラーが発生しました: {e}"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text.strip()
    
    # 4桁の数字（銘柄コード）が送られた場合
    if re.match(r'^\d{4}$', user_text):
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"【{user_text}】を詳細分析中です...少々お待ちください。")
        )
        report = analyze_stock(user_text)
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=report)
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="分析したい4桁の銘柄コード（例：7203）を送信してください。")
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
