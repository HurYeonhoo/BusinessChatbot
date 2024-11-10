import openai
import pandas as pd

def main():
    openai.api_key = 'api_key' 

    positive_keywords = {
        "최고": 3,
        "추천": 3,
        "진짜 좋다": 2.5,
        "대박": 3,
        "극찬": 3,
        "아늑": 2,
        "좋았습니다": 2,
        "맛집": 3,
        "진리": 3,
        "핫플": 2.5,
        "맛있": 2.5,
        "오고 싶다": 2.5,
        "너무 친절해서": 2,
        "기분 좋게": 1.5,
        "분위기도 좋고": 1.5,
        "진짜 ㄹㅇ": 2,
        "맛있어요": 2,
        "좋은": 1,
        "왔는데": 1,
        "정말": 1.5,
        "좋아요": 1
    }

    # 부정적인 핵심어 및 n-grams와 가중치
    negative_keywords = {
        "실망": 3,
        "아쉽다": 2.5,
        "그냥 그래": 2.5,
        "실망스러움": 3,
        "음식이 너무": 2,
        "너무 짜서": 1.5,
        "그냥 그랬어요": 2,
        "기대 이하였습니다": 2,
        "늦게 나오고": 1.5,
        "별로였어요": 2,
        "이렇게": 1,
        "너무": 1
    }

    # 주어진 input이 긍정이면 1, 부정이면 0을 결과로 내라.
    system_prompt = f"""
    당신은 고객 리뷰의 감정을 정확하게 분류하는 전문가입니다. 주어진 리뷰를 신중히 분석하여 그 감정을 판단하세요.

    다음 지침을 따라주세요:
    1. 리뷰가 전반적으로 긍정적이면 '1'을 출력하세요.
    2. 리뷰가 전반적으로 부정적이거나 중요한 부정적 요소를 포함하고 있다면 '0'을 출력하세요.
    3. 반드시 정수 '1' 또는 '0'만을 출력하고, 다른 텍스트는 포함하지 마세요.

    분류 시 다음 사항을 고려하세요:
    - 리뷰의 전반적인 톤과 감정을 주의 깊게 파악하세요.
    - 긍정적인 표현과 부정적인 표현의 비중을 신중히 비교하세요.
    - 그러나 중요한 부분에서 심각한 불만이나 문제점을 지적하고 있다면 부정으로 분류하세요.

    부정 리뷰 판단 시 특히 주의해야 할 점:
    - 서비스 품질, 음식 맛, 위생 상태, 가게 분위기 등 핵심적인 요소에 대한 불만이 있는지 확인하세요.
    - 고객이 재방문 의사를 부정적으로 표현하고 있는지 주목하세요.
    - 강한 부정적 감정을 나타내는 단어나 표현이 사용되었는지 살펴보세요.

    분류에 어려움이 있다면, 다음 키워드 목록을 참조할 수 있습니다:
    긍정 키워드: {positive_keywords}
    부정 키워드: {negative_keywords}

    리뷰를 신중히 분석하고, 균형 잡힌 판단을 내려주세요. 부정적인 요소에 특별히 주의를 기울이되, 전체적인 맥락을 고려하여 분류해주세요.
    """

    # 파일을 DataFrame으로 로드 # 크롤링한 파일 입력
    df = pd.read_csv(r"gpt_contest\reviews.csv")
    df.drop(columns=['Unnamed: 0']) # 전처리는 데이터 형식에 따라 진행
    df = df.rename(columns={'review': 'comment'})
    df = df.dropna()

    predicted_labels = []

    total = len(df)

    # 각 row를 순회하며 코멘트를 분류
    for index, row in df.iterrows():
        comment = row['comment']
        predicted_label = classify_text(comment, system_prompt)
        predicted_labels.append(predicted_label)

        print(f"[{index+1}]/[{total}]")
        print("comment : ", comment)
        print("predicted class : ", predicted_label)
        print("---------------")

        # if index > 3:
        #     break

    # 예측된 레이블을 DataFrame에 추가
    df['label'] = predicted_labels

    file_path = r"gpt_contest\review_labeled.csv"
    df[['comment', 'label']].to_csv(file_path, sep='\t', index=False)

def llm(input_text, system_prompt):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
    )
    response_message = completion.choices[0].message.content

    # "1입니다" 라고 나올 수 있음
    if '1' in response_message:
        return 1
    else:
        return 0
    
def classify_text(input_text, system_prompt):
    output = llm(input_text, system_prompt)
    return output

if "__name__" == "__name__":
    main()
