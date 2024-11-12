import pandas as pd
import re
from konlpy.tag import Mecab, Okt
from collections import Counter
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud

url = "https://raw.githubusercontent.com/byungjooyoo/Dataset/main/korean_stopwords.txt"
response = requests.get(url)

mpl.font_manager.fontManager.addfont(r'data\NanumGothic-Bold.ttf')
font_path = r'data\NanumGothic-Bold.ttf'

data = pd.read_csv(r'data\review_labeled.csv', sep='\t')

# 라벨링된 리뷰를 분류
positive_data = data[data['label'] == 1].copy()  
negative_data = data[data['label'] != 1].copy() 

# 필요없는 문자 제거
pattern = re.compile(r'[가-힣\s]+')

positive_data['ko_text'] = positive_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))
negative_data['ko_text'] = negative_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))

# 문자열 하나로 병합
positive_ko_text = ','.join(positive_data['ko_text'].dropna())
negative_ko_text = ','.join(negative_data['ko_text'].dropna())

# 형태소 분류
okt = Okt()
positive_pos = okt.pos(positive_ko_text)
negative_pos = okt.pos(negative_ko_text)

# 명사 추출
positive_nouns = okt.nouns(positive_ko_text)
negative_nouns = okt.nouns(negative_ko_text)

# 길이 2 이상 추출
positive_nouns_result = [n for n in positive_nouns if len(n) > 1]
negative_nouns_result = [n for n in negative_nouns if len(n) > 1]

# # 최빈 30개 추출
# positive_count = Counter(positive_nouns_result)
# positive_top = positive_count.most_common(30)

# negative_count = Counter(negative_nouns_result)
# negative_top = negative_count.most_common(30)

# 불용어 읽기
with open('korean_stopwords.txt', 'w') as f:
    f.write(response.text)

with open('korean_stopwords.txt', 'r') as f:
    stop_words = f.read().split("\n")

# 불용어 제거
positive_corpus = [x for x in positive_nouns_result if x not in stop_words]
negative_corpus = [x for x in negative_nouns_result if x not in stop_words]


# 최빈 30개 추출
positive_count = Counter(positive_corpus).most_common(30)
negative_count = Counter(negative_corpus).most_common(30)

# 중복 단어 처리: 긍정과 부정에서 중복된 단어는 부정 단어에만 _neg 접미사 추가
combined_freq = {}
positive_words = set(word for word, _ in positive_count)
for word, freq in positive_count:
    combined_freq[word] = freq  # 긍정 단어는 그대로 추가

for word, freq in negative_count:
    if word in positive_words:
        combined_freq[f"{word}_neg"] = freq  # 중복된 경우 부정 단어에만 _neg 추가
    else:
        combined_freq[word] = freq  # 중복되지 않은 부정 단어는 그대로 추가

# 색상을 긍정, 부정에 따라 다르게 지정하는 함수
def color_func(word, *args, **kwargs):
    if word in dict(positive_count):
        return 'blue'
    elif word in dict(negative_count):
        return 'red'
    else:
        return 'red'

# 워드 클라우드 생성
wc = WordCloud(
    font_path=font_path,
    background_color='white',
    width=800,
    height=600,
    color_func=color_func
).generate_from_frequencies(combined_freq)

# 시각화 및 저장
fig = plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')

# 이미지 저장
plt.savefig('combined_wordcloud.png', format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# # 긍정 단어 출력
# wc = WordCloud(
#     font_path=font_path, 
#     background_color='black',
#     width=800,
#     height=600
# ).generate(positive_text)

# fig = plt.figure(figsize=(10, 10))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# # 부정 단어 출력
# wc = WordCloud(
#     font_path=font_path, 
#     background_color='black',
#     width=800,
#     height=600
# ).generate(positive_text)

# fig = plt.figure(figsize=(10, 10))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()
