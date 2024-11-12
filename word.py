import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

class SentimentWordCloud:
    def __init__(self, review_labels):
        self.review_labels = review_labels

    def generate_wordcloud(self):
        # 불용어 다운로드
        url = "https://raw.githubusercontent.com/byungjooyoo/Dataset/main/korean_stopwords.txt"
        response = requests.get(url)
        stop_words = response.text.split("\n")
        
        # 폰트 설정
        mpl.font_manager.fontManager.addfont('data/NanumGothic-Bold.ttf')

        positive_data = self.review_labels[self.review_labels['label'] == 1].copy()
        negative_data = self.review_labels[self.review_labels['label'] != 1].copy()
        
        # 필요 없는 문자 제거
        pattern = re.compile(r'[가-힣\s]+')
        positive_data['ko_text'] = positive_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))
        negative_data['ko_text'] = negative_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))

        # 문자열 하나로 병합
        positive_ko_text = ','.join(positive_data['ko_text'].dropna())
        negative_ko_text = ','.join(negative_data['ko_text'].dropna())

        # 형태소 분류
        okt = Okt()
        positive_nouns = [n for n in okt.nouns(positive_ko_text) if len(n) > 1 and n not in stop_words]
        negative_nouns = [n for n in okt.nouns(negative_ko_text) if len(n) > 1 and n not in stop_words]

        # 명사 추출
        positive_count = Counter(positive_nouns).most_common(30)
        negative_count = Counter(negative_nouns).most_common(30)
        
        # 길이 2 이상 추출
        combined_freq = {}
        positive_words = set(word for word, _ in positive_count)
        for word, freq in positive_count:
            combined_freq[word] = freq
        for word, freq in negative_count:
            if word in positive_words:
                combined_freq[f"{word}_neg"] = freq
            else:
                combined_freq[word] = freq

        # 색상을 긍정/부정에 따라 지정하는 함수
        def color_func(word, *args, **kwargs):
            if word in dict(positive_count):
                return 'blue'
            elif word in dict(negative_count):
                return 'red'
            else:
                return 'red'

        # 워드 클라우드 생성
        wc = WordCloud(
            font_path='data/NanumGothic-Bold.ttf',
            background_color='white',
            width=800,
            height=600,
            color_func=color_func
        ).generate_from_frequencies(combined_freq)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        # 이미지 데이터를 반환
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', transparent=True)
        img_buffer.seek(0)
        return img_buffer.getvalue()