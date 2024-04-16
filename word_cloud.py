import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
plt.show()