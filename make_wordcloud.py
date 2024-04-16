from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from openai import OpenAI
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt

browser = webdriver.Chrome()

search = input("검색어를 입력하세요: ")

url = f"https://arxiv.org/search/?searchtype=all&query={search}&abstracts=show&size=100&order=-announced_date_first"

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
    # 테스트로 3개만 해봄
    # if i == 3:
    #     break

print(len(list))

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