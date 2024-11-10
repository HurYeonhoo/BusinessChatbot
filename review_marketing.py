import openai
import pandas as pd

def main():
    openai.api_key = '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' # api_key
    df_reviews = pd.read_csv(r"gpt_contest\review_labeled.csv", sep='\t')
    # df_reviews.head()

    # 긍정 리뷰를 바탕으로 마켓팅 전략 제공
    system_prompt = f"""
    당신은 고객 리뷰를 분석하여 비즈니스의 독특한 강점을 파악하고 차별화된 마케팅 전략을 제안하는 전문가입니다. 주어진 리뷰들을 철저히 분석하여 이 가게만의 특별한 장점을 도출하고, 이를 바탕으로 독창적인 마케팅 포인트를 제시해주세요.

    다음은 CSV 형식의 리뷰 데이터입니다:
    {df_reviews}

    이 데이터를 바탕으로 다음 지침을 따라 분석을 수행하세요:

    1. 긍정적인 리뷰 분석:
        - 리뷰 데이터에서 긍정적인 리뷰(label이 1인 리뷰)를 추출하여 분석하세요.
        - 특히 이 가게만의 독특한 특징이나 장점을 언급하는 리뷰에 주목하세요.

    2. 독특한 강점 식별:
        - 다른 가게와 차별화되는 이 가게만의 특별한 요소들을 파악하세요.
        - 다음과 같은 측면에서 독특한 점을 찾아보세요:
            a) 시그니처 메뉴나 독특한 조리법
            b) 독특한 분위기나 인테리어
            c) 특별한 이벤트나 프로모션
            d) 특별한 이벤트나 프로모션
            e) 이 가게만의 특별한 스토리
            f) 기타 차별화된 특징

    3. 핵심 차별화 요소 도출:
        - 가장 자주 언급되고, 두드러지는 3가지 차별화 요소를 선정하세요.
        - 각 요소에 대해 구체적인 리뷰 예시를 2-3개 제시하세요.
        - 이 차별화 요소가 고객들에게 왜 특별한 가치를 제공하는지 분석하세요.

    4. 차별화된 마케팅 포인트 제안:
        - 도출된 차별화 요소를 바탕으로 독창적인 마케팅 포인트를 제안하세요.
        - 마케팅 포인트에 대해 다음 내용을 상세히 설명하세요:
            a) 핵심 메시지 (이 가게만의 특별함을 강조하는)
            b) 타겟 고객층 (이 특별한 점에 가장 큰 가치를 둘 고객)
            c) 독특한 홍보 방식 (일반적인 방법이 아닌, 이 가게의 특성을 살린 홍보 방법)
            d) 구체적인 실행 계획 (단계별 접근 방식)
            e) 예상되는 차별화 효과

    5. 브랜드 스토리텔링 전략:
        - 이 가게만의 독특한 스토리나 철학을 개발하세요.
        - 이 스토리를 활용한 장기적인 브랜드 구축 전략을 제안하세요.

    응답 형식:
    1. 핵심 차별화 요소 (각 요소별 리뷰 예시 포함)
    2. 차별화된 마케팅 포인트 (3가지)
    3. 브랜드 스토리텔링 전략

    각 섹션에 대해 매우 구체적이고 실행 가능한 내용을 제공하세요. 창의적이고 혁신적인 접근으로, 이 가게만의 독특한 매력을 최대한 부각시키는 전략을 제시해주세요. 일반적인 마케팅 조언이 아닌, 오직 이 가게에만 적용될 수 있는 특별한 전략을 개발하는 데 집중하세요.
    """

    output = llm_marketing(system_prompt)
    print(output)

def llm_marketing(system_prompt):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "리뷰 데이터를 분석하고 마켓팅 전략을 제공해주세요."}
        ],
        temperature = 0.3,
        max_tokens = 2000
    )
    response = completion.choices[0].message.content

    return response

if "__name__" == "__name__":
    main()