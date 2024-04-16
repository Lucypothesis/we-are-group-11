from openai import OpenAI
import openai
import pandas as pd

csv = pd.read_csv("./arxiv_crawling.csv", sep = ',')
df = pd.DataFrame(csv)
abstracts = csv['초록'].tolist()

keywords2 = []
one_line2 = []


client = OpenAI()

openai.api_key = "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"
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



