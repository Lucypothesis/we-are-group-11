from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from st_audiorec import st_audiorec
# from audiorecorder import audiorecorder
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from PIL import Image
import base64
import os
from gtts import gTTS
import plotly.graph_objects as go
import json
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import requests



def main():
    st.set_page_config(
        page_title="Articlend",
        layout="wide")

    # session state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []
    
    st.header("트렌드를 분석하는 <11조>입니다.💫")

    # 사용자 입력 받아서 워드클라우드 만들기
    st.sidebar.title("어떤 트렌드가 궁금하신가요?")
    search = st.sidebar.text_input("아래 칸에 검색어를 입력하세요")
    search_date = st.sidebar.date_input("검색을 시작할 날짜를 입력하세요(yyyy-mm-dd)", value = pd.to_datetime("2023-01-01"))
    search_db = st.sidebar.selectbox("검색할 데이터베이스를 선택하세요", ["arXiv", "jstor"])
    search_button = st.sidebar.button("검색 시작")
    today = time.strftime('%Y/%m/%d', time.localtime(time.time()))
    
    img1 = Image.open('arti.png')
    st.sidebar.image(img1,width=300)
    
    if search_button:
#----------------------------------
        # 사용 예제
# # ---------------------------------
        # search = input("검색어를 입력하세요: ")
        browser = webdriver.Chrome()
        if search_db == "arXiv":
        
          url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={search_keyword}&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={search_date}&date-to_date=2024-04-15&date-date_type=submitted_date&abstracts=show&size=100&order=-announced_date_first"
          find_container = 'list-title'
          find_title = 'title'
          find_author = 'authors'
          find_abstract = 'abstract'
          find_date = 'submission-history'
          
        else:
          search_keyword = search_keyword.replace(" ", "+")
          search_date = search_date.replace("-", "/")
          url = f"https://www.jstor.org/action/doAdvancedSearch?q0={search_keyword}&sd={search_date}&ed={today}&f0=all&c1=AND&f1=all&acc=on&so=rel"
          find_container = 'result__main__metadata'
          find_title = 'title'
          find_author = 'search-results-vue-pharos-link'
          find_author2 = 'contrib metadata-row'
          find_date = 'break-word'
          
        browser.get(url)
        # search 값 초기화
        st.session_state['search'] = ""
        
  
        list = []
        wait = WebDriverWait(browser, 5)
        links = WebDriverWait(browser, 20).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, find_container)))
        st.write(f'{len(links)}개의 논문이 검색되었습니다.')
        i = 0
        for link in links:
          if search_db == "arXiv":
            title = WebDriverWait(browser, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, find_title))).text
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, find_author)))
            abstract = browser.find_element(By.CLASS_NAME,find_abstract).text
            author = browser.find_element(By.CLASS_NAME,find_author).text
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, find_date)))
            date = browser.find_element(By.CLASS_NAME, find_date).text
          else:
          
            try:
              browser.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + 100;', links)
              time.sleep(1)
              title = link.find_element(By.CLASS_NAME, find_title).text
              print(f'{title}+ "제목입니다"')
              time.sleep(1)
              abstract = "JSTOR에서는 초록을 제공하지 않습니다."
              author = "JSTOR에서는 저자를 제공하지 않습니다."
              # author_link = browser.find_element(By.CLASS_NAME, find_author2).find_element(By.TAG_NAME,find_author).get_attribute('href')
              # author = author_link.split('=')[-1].replace('+', ' ')
              time.sleep(1)
              wait.until(EC.presence_of_element_located((By.CLASS_NAME, find_date)))
              date = link.find_element(By.CLASS_NAME, find_date).find_element(By.CLASS_NAME, 'metadata').text
              print(f'{date}+ "게재일입니다"')
              browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            except:
              title = "제목을 찾을 수 없습니다."
              date = "게재일을 찾을 수 없습니다."
          
          
          
          list.append(
            {'제목': title,
            '저자': author,
            '초록': abstract,
            '게재일': date}
          )
          # browser.back()
          # WebDriverWait(browser, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, find_container)))
          i += 1
          print(i,'개 추출 성공')
          # 테스트로 3개만 해봄
          # if i == 3:
          #     break

        browser.quit()

        df = pd.DataFrame(list)
        df.to_csv("crawl.csv",encoding='utf-8-sig')
        print('crawl.csv 파일 생성')
        print('한줄 요약 생성중')

        # Download the CSV file
        st.download_button(
          label="Download CSV",
          data=df.to_csv().encode('utf-8-sig'),
          file_name="crawl.csv",
          mime="text/csv"
        )

        df = pd.read_csv('crawl.csv')
        st.dataframe(df)
        

        #########################################################
        csv = pd.read_csv("./crawl.csv", sep = ',')
        df = pd.DataFrame(csv)
        abstracts = csv['초록'].tolist()

        keywords2 = []
        one_line2 = []
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        for abstract in abstracts:
            response = openai.ChatCompletion.create(
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

        df.to_csv('crawl.csv',encoding='utf-8-sig')
        print('한줄 요약 파일 추출 성공')
        print('워드클라우드 만드는 중')
        #####################################################
        # CSV 파일 경로
        file_path = r"crawl.csv"

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
        plt.figure(figsize = (15, 15), facecolor = None) 
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        # 이미지로 저장
        plt.savefig('word_cloud.png')
        print('워드클라우드 만들기 성공')
    st.session_state['search'] = ""
    
    


    def send_slack_message(message):
        hook_url = "https://hooks.slack.com/services/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" 
        headers = {'Content-type': 'application/json'}
        data = {
            "text": message
        }
        response = requests.post(hook_url, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"Slack request returned an error {response.status_code}, the response is:\n{response.text}")

    def format_bestseller_message(index, row):

        return f"번호: {index + 1}\n제목: {row['제목']}\n저자: {row['저자']}\n초록: {row['초록']}\n게재일: {row['게재일']}\n한줄요약: {row['한 줄 요약']}\n키워드: {row['키워드']}"

    
    df = pd.read_csv('crawl.csv')

    # Streamlit app
    st.title("슬랙으로 보내기")
    button_clicked = st.button("전송 시작")
    if button_clicked:
        # 전체 데이터셋 순회
        st.write("전송 중...예상 시간 1초/개")
        for index, row in df.iterrows():
            message = format_bestseller_message(index, row)
            send_slack_message(message)
            progress_message = f"진행 중... ({index+1}/{len(df)})"  # 진행 상황 메시지 생성
            st.write(progress_message)
            time.sleep(1)
        st.write("전송 완료! ")

#----------------------------------
    # flag_start = False
    
    # openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    # def STT(audio):
    #     filename='input.mp3'
    #     wav_file = open(filename, "wb")
    #     wav_file.write(audio.tobytes())
    #     wav_file.close()
    
    #     # 음원 파일 열기
    #     audio_file = open(filename, "rb")
    #     # Whisper 적용!!!
    #     transcript = openai.Audio.transcribe("whisper-1", audio_file)
    #     audio_file.close()
    #     # 파일 삭제
    #     os.remove(filename)
    #     return transcript["text"]
    
    # def ask_gpt(prompt):
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=prompt)
    #     return response.choices[0].message['content']
    
    # def TTS(response):
    #     # gTTS 를 활용하여 음성 파일 생성
    #     filename = "output.mp3"
    #     tts = gTTS(text=response,lang="ko")
    #     tts.save(filename)
    
    #     # 음원 파일 자동 재생
    #     with open(filename, "rb") as f:
    #         data = f.read()
    #         b64 = base64.b64encode(data).decode()
    #         md = f"""
    #             <audio autoplay="True">
    #             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    #             </audio>
    #             """
    #         st.markdown(md,unsafe_allow_html=True,)
    #     # 파일 삭제
    #     os.remove(filename)
    # ---------------------------------------
    def process_csv(file):
        df = pd.read_csv(file)
        summary = df.describe().to_string()  # DataFrame을 문자열로 변환
        return summary
    # ---------------------------------------
    file_path = 'word_cloud.png'
    if os.path.exists(file_path):
            st.success('Done!')
    ### 칼럼
    col1, col2, col3 =  st.columns([3,3,3])
    with col1:
        st.header("Word Cloud")
        file_path = 'word_cloud.png'
        if os.path.exists(file_path):
            img2 = Image.open(file_path)
            st.image(img2, width=350)
    # with col2:
    #     st.subheader("어떤 것이 궁금한가요?")
    #     # 음성 녹음 아이콘
        
    #     audio = audiorecorder("🐣여기를 클릭하여 말하십쇼~🐣", "👾말하기가 끝나면 누르십쇼~👾")
    #     if len(audio) > 0 and not np.array_equal(audio,st.session_state["check_audio"]):
    #         # 음성 재생 
    #         st.audio(audio.tobytes())
    #         # 음원 파일에서 텍스트 추출
    #         question = STT(audio)
    #         # 채팅을 시각화하기 위해 질문 내용 저장
    #         now = datetime.now().strftime("%H:%M")
    #         st.session_state["chat"] = st.session_state["chat"]+ [("user",now, question)]
    #         # GPT 모델에 넣을 프롬프트를 위해 질문 내용 저장
    #         st.session_state["messages"] = st.session_state["messages"]+ [{"role": "user", "content": question}]
    #         # audio 버퍼 확인을 위해 현 시점 오디오 정보 저장
    #         st.session_state["check_audio"] = audio
    #         flag_start =True

    # with col3:
    #     st.subheader("질문/답변")
    #     if flag_start:
    #         #ChatGPT에게 답변 얻기
    #         response = ask_gpt(st.session_state["messages"])

    #         # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
    #         st.session_state["messages"] = st.session_state["messages"]+ [{"role": "system", "content": response}]

    #         # 채팅 시각화를 위한 답변 내용 저장
    #         now = datetime.now().strftime("%H:%M")
    #         prompt = [{"role": "system", "content": "You are an analytical assistant capable of understanding detailed CSV data."},
    #                 {"role": "user", "content": process_csv('arxiv_crawling.csv')},
    #                 {"role": "user", "content": question}]
    #         response = ask_gpt(prompt)
    #         st.session_state["chat"].append(("bot", now, response))

    #         # 채팅 형식으로 시각화 하기
    #         for sender, Time, message in st.session_state["chat"]:
    #             if sender == "user":
    #                 st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{Time}</div></div>', unsafe_allow_html=True)
    #                 st.write("")
    #             else:
    #                 st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{Time}</div></div>', unsafe_allow_html=True)
    #                 st.write("")
            
    #         # gTTS 를 활용하여 음성 파일 생성 및 재생
    #         TTS(response)
    # ---------------------------------
    st.markdown("---")

    ### 사이드바

    # st.sidebar.title("주식 데이터 시각화")
    # ticker = st.sidebar.text_input("ticker를 입력하세요 (e. g. AAPL)", value = "AAPL")
    # st.sidebar.markdown('ticker 출처 : [All Stock Symbols](https://stockanalysis.com/stocks/)')
    # start_date = st.sidebar.date_input("시작 날짜: ", value = pd.to_datetime("2023-01-01"))
    # end_date = st.sidebar.date_input("종료 날짜: ", value = pd.to_datetime("2023-07-28"))

    # # ticker 종목의 시작~종료 날짜 사이의 가격변화를 데이터로 보여줌
    # data = yf.download(ticker, start= start_date, end= end_date)
    # st.dataframe(data)

    # # Line Chart, Candle Stick 중 선택
    # chart_type = st.sidebar.radio("Select Chart Type", ("Candle_Stick", "Line"))
    # candlestick = go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])
    # line = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')

    # if chart_type == "Candle_Stick":
    #     fig = go.Figure(candlestick)
    # elif chart_type == "Line":
    #     fig = go.Figure(line)
    # else:
    #     st.error("error")

    # fig.update_layout(title=f"{ticker} 주식 {chart_type} 차트", xaxis_title="Date", yaxis_title="Price")
    # st.plotly_chart(fig)

    # ### 데이터셋
    # iris_dataset = load_iris()

    # df= pd.DataFrame(data=iris_dataset.data,columns= iris_dataset.feature_names)
    # df.columns= [ col_name.split(' (cm)')[0] for col_name in df.columns] # 컬럼명을 뒤에 cm 제거하였습니다
    # df['species']= iris_dataset.target 
    
    
    # species_dict = {0 :'setosa', 1 :'versicolor', 2 :'virginica'} 
    
    # def mapp_species(x):
    #   return species_dict[x]
    
    # df['species'] = df['species'].apply(mapp_species)
    
    # #####
    # st.sidebar.markdown("---")
    # st.sidebar.title('Select Species🌸')
    
    # select_species = st.sidebar.selectbox(
    #     '확인하고 싶은 종을 선택하세요',
    #     ['setosa','versicolor','virginica']
    # )
    # tmp_df = df[df['species']== select_species]
    # st.table(tmp_df.head())
    # st.sidebar.markdown("---")

    # with st.sidebar:
    #     st.subheader("체크박스들")
    #     st.checkbox("checkbox1")
    #     st.checkbox("checkbox2")
    #     st.markdown("---")
        
    # # 슬라이더 추가
    # with st.sidebar:
    #     value2 = st.slider("숫자를 선택하세요",0, 100)
    #     st.write(value2)

    ### Display data & Chart
    col4, col5 =  st.columns([2,3])
    with col4:
        st.header('This is a data frame.')
        dataframe = pd.DataFrame(np.random.randn(10, 5),
        columns = ('col %d' % i for i in range(5)))
        dataframe
    with col5:
        st.header('This is a line chart.')
        st.line_chart(dataframe)
    

if __name__=="__main__":
    main()