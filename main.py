import streamlit as st
from audiorecorder import audiorecorder
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from PIL import Image
import openai
import base64
import os
from gtts import gTTS
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from sklearn.datasets import load_iris 

def main():
    st.set_page_config(
        page_title="we-are-crawling-the-trends",
        layout="wide")

    # session state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []
    
    st.header("트렌드를 분석하는 <11조>입니다.💫")
    st.markdown("---")

    # 사용자 입력 받기
    search_query = st.text_input("검색어를 입력하세요:")
    
    # 검색 버튼
    if st.button('검색'):
        if search_query:
            browser = webdriver.Chrome()
            # 구성된 URL로 이동
            url = f"https://arxiv.org/search/?searchtype=all&query={search_query}&abstracts=show&size=100&order=-announced_date_first"
            browser.get(url)
            list = []

            links = WebDriverWait(browser, 20).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, 'list-title')))
            
            i = 0
            for link in links:
                link.click()
                title = WebDriverWait(browser, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, 'title'))).text
                author = browser.find_element(By.CLASS_NAME,'authors').text.split(', ')
                abstract = browser.find_element(By.CLASS_NAME,'abstract').text
                date = browser.find_element(By.CLASS_NAME,'submission-history').text.split('\n')[2][5:34]
                list.append(
                    {'제목': title,
                     '저자': author,
                     '초록': abstract,
                     '게재일': date}
                )
                browser.back()
                WebDriverWait(browser, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, 'list-title')))
                i += 1
                print(i,'개 추출 성공')
            
            browser.quit()
            
            df = pd.DataFrame(list)
            df.to_csv("arxiv_crawl.csv",encoding='utf-8-sig')
            print('arxiv_crawl csv 파일 추출 성공')
            print('한줄 요약 csv 파일 추출중')
            #########################################################
            csv = pd.read_csv("./arxiv_crawl.csv", sep = ',')
            df = pd.DataFrame(csv)
            abstracts = csv['초록'].tolist()
            
            keywords2 = []
            one_line2 = []
            
            client = OpenAI(api_key="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")
            
            # openai.api_key = "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"
            for abstract in abstracts:
              response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                      {"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Extract keywords and provide a one-sentence summary of the following abstract in korean:\n\n{abstract}"}
                  ],
                temperature=0.5,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
              )
            
              # 응답에서 텍스트 내용 추출
              content = response.choices[0].message.content
            
              # 'Keywords' 부분과 '한 문장 요약' 부분 분리
              keyword_start = content.find("Keywords:") + len("Keywords: ")
              summary_start = content.find("한 문장 요약:") + len("한 문장 요약: ")
            
              # 각 섹션의 끝 찾기
              keyword_end = content.find("\n\n", keyword_start)
              summary_end = len(content)
            
              # 키워드와 요약 텍스트 변수에 저장
              keywords = content[keyword_start:keyword_end].strip().split(', ')
              summary = content[summary_start:summary_end].strip()
            
              keywords2.append(keywords)
              one_line2.append(summary)
            
            df['한 줄 요약'] = one_line2
            df['키워드'] = keywords2
            
            df.to_csv('arxiv_crawling.csv',encoding='utf-8-sig')
            print('한줄 요약 파일 추출 성공')
            print('워드클라우드 만드는 중')
            #####################################################
            # CSV 파일 경로
            file_path = r"arxiv_crawling.csv"
            
            # CSV 파일 읽기
            data = pd.read_csv(file_path)
            
            # 워드 클라우드 생성을 위한 텍스트 데이터 추출
            text = ' '.join(data['초록'].dropna())  # NaN 값 제외
            
            # 워드 클라우드 객체 생성
            wordcloud = WordCloud(width = 800, height = 800, 
                                  background_color ='white', 
                                  stopwords = None, 
                                  min_font_size = 10).generate(text)
            
            # 워드 클라우드 시각화
            plt.figure(figsize = (8, 8), facecolor = None) 
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
            
            # 이미지로 저장
            plt.savefig('arxiv_word_cloud.png')
            print('워드클라우드 만들기 성공')
            plt.show()
---------------------------------

    ### 칼럼
    
    flag_start = False
    
    openai.api_key = os.environ['OPENAI_KEY']
    
    def STT(audio):
        filename='input.mp3'
        wav_file = open(filename, "wb")
        wav_file.write(audio.tobytes())
        wav_file.close()
    
        # 음원 파일 열기
        audio_file = open(filename, "rb")
        # Whisper 적용!!!
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        audio_file.close()
        # 파일 삭제
        os.remove(filename)
        return transcript["text"]
    
    def ask_gpt(prompt, model):
        response = openai.ChatCompletion.create(model=model, messages=prompt)
        system_message = response["choices"][0]["message"]
        return system_message["content"]
    
    def TTS(response):
        # gTTS 를 활용하여 음성 파일 생성
        filename = "output.mp3"
        tts = gTTS(text=response,lang="ko")
        tts.save(filename)
    
        # 음원 파일 자동 재생
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="True">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md,unsafe_allow_html=True,)
        # 파일 삭제
        os.remove(filename)

    col1, col2 =  st.columns([3,5])
    with col1:
        st.subheader("어떤 것이 궁금한가요?")
        # 음성 녹음 아이콘
        audio = audiorecorder("🐣여기를 클릭하여 말하십쇼~🐣", "👾말하기가 끝나면 누르십쇼~👾")
        if len(audio) > 0 and not np.array_equal(audio,st.session_state["check_audio"]):
            # 음성 재생 
            st.audio(audio.tobytes())

            # 음원 파일에서 텍스트 추출
            question = STT(audio)
            # 채팅을 시각화하기 위해 질문 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"]+ [("user",now, question)]
            # GPT 모델에 넣을 프롬프트를 위해 질문 내용 저장
            st.session_state["messages"] = st.session_state["messages"]+ [{"role": "user", "content": question}]
            # audio 버퍼 확인을 위해 현 시점 오디오 정보 저장
            st.session_state["check_audio"] = audio
            flag_start =True
        img1 = Image.open('tell-me.jpg')
        st.image(img1,width=200)

    with col2:
        st.subheader("질문/답변")
        if flag_start:
            #ChatGPT에게 답변 얻기
            response = ask_gpt(st.session_state["messages"], "gpt-3.5-turbo")

            # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
            st.session_state["messages"] = st.session_state["messages"]+ [{"role": "system", "content": response}]

            # 채팅 시각화를 위한 답변 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"]+ [("bot",now, response)]

            # 채팅 형식으로 시각화 하기
            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
                else:
                    st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
            
            # gTTS 를 활용하여 음성 파일 생성 및 재생
            TTS(response)
    st.markdown("---")

    ### 사이드바

    st.sidebar.title("주식 데이터 시각화")
    ticker = st.sidebar.text_input("ticker를 입력하세요 (e. g. AAPL)", value = "AAPL")
    st.sidebar.markdown('ticker 출처 : [All Stock Symbols](https://stockanalysis.com/stocks/)')
    start_date = st.sidebar.date_input("시작 날짜: ", value = pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("종료 날짜: ", value = pd.to_datetime("2023-07-28"))

    # ticker 종목의 시작~종료 날짜 사이의 가격변화를 데이터로 보여줌
    data = yf.download(ticker, start= start_date, end= end_date)
    st.dataframe(data)

    # Line Chart, Candle Stick 중 선택
    chart_type = st.sidebar.radio("Select Chart Type", ("Candle_Stick", "Line"))
    candlestick = go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])
    line = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')

    if chart_type == "Candle_Stick":
        fig = go.Figure(candlestick)
    elif chart_type == "Line":
        fig = go.Figure(line)
    else:
        st.error("error")

    fig.update_layout(title=f"{ticker} 주식 {chart_type} 차트", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    ### 데이터셋
    iris_dataset = load_iris()

    df= pd.DataFrame(data=iris_dataset.data,columns= iris_dataset.feature_names)
    df.columns= [ col_name.split(' (cm)')[0] for col_name in df.columns] # 컬럼명을 뒤에 cm 제거하였습니다
    df['species']= iris_dataset.target 
    
    
    species_dict = {0 :'setosa', 1 :'versicolor', 2 :'virginica'} 
    
    def mapp_species(x):
      return species_dict[x]
    
    df['species'] = df['species'].apply(mapp_species)
    
    #####
    st.sidebar.markdown("---")
    st.sidebar.title('Select Species🌸')
    
    select_species = st.sidebar.selectbox(
        '확인하고 싶은 종을 선택하세요',
        ['setosa','versicolor','virginica']
    )
    tmp_df = df[df['species']== select_species]
    st.table(tmp_df.head())
    st.sidebar.markdown("---")

    with st.sidebar:
        st.subheader("체크박스들")
        st.checkbox("checkbox1")
        st.checkbox("checkbox2")
        st.markdown("---")
        
    # 슬라이더 추가
    with st.sidebar:
        value2 = st.slider("숫자를 선택하세요",0, 100)
        st.write(value2)

if __name__=="__main__":
    main()
