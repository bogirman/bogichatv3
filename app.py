# encoding: utf-8
from flask import Flask, request, abort

from linebot import(
    LineBotApi, WebhookHandler
)

from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
import jieba
import jieba.analyse

app = Flask(__name__)

line_bot_api = LineBotApi('OewxcVOvTqu19DlPknIfBKJDcbf15a1hipzDw6/teEUAOleQqWvHm0IG0FGJrK5XnKqY0CsW/V10ZRkzheqMQ2UtNxl0cTAR92FZ/+vl6VqL8ir6iEuP0D71x4hoHmMYqjpqOwUp88/5GkadfGWXBAdB04t89/1O/w1cDnyilFU=') #Your Channel Access Token
handler = WebhookHandler('fdcd90a85a349704e31dfd3d6e6e16c2') #Your Channel Secret

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    jieba.set_dictionary('jieba/dict.txt.big')
    text = event.message.text #message from user
    textforanalyse = text
    #一般斷詞
    #tags = jieba.cut(text, cut_all=False)
    #TextToUser = " "
    #for word in tags:
        #if word is str:
    #    TextToUser +=',' + word
    #    num = num + 1
    #line_bot_api.reply_message(
    #    event.reply_token,
    #    TextSendMessage(text='分詞：'+TextToUser)) #reply the same message from user
    #提取關鍵字
    tags = jieba.analyse.extract_tags(textforanalyse,5)
    TextToUser = " "
    for word in tags:
        #if word is str:
        TextToUser +=',' + word
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='關鍵字提取：'+TextToUser)) #reply the same message from user

import os
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=os.environ['PORT'])