import openai
import pandas as pd

def main():
    openai.api_key = 'api_key'
    df_reviews = pd.read_csv(r"gpt_contest\review_labeled.csv", sep='\t')
    # df_reviews.head()

    # 부정 리뷰를 바탕으로 개선 방안 제공 
    system_prompt = f"""
    당신은 고객 피드백 분석 전문가입니다. 0(부정)과 1(긍정)로 구분된 리뷰 데이터 중 부정적인 리뷰를 분석하여 가게에 대한 건설적인 피드백과 구체적인 개선 방안을 제시해야 합니다.

    다음은 CSV 형식의 리뷰 데이터입니다:
    {df_reviews}

    이 데이터를 바탕으로 다음 작업을 수행하세요:

    1. 부정적인 리뷰(0으로 표시된 리뷰)만을 추출하여 분석하세요.

    2. 부정적인 리뷰에서 언급되는 모든 문제점을 식별하고, 다음 예시와 같이 비슷한 리뷰들을 각 카테고리로 분류하세요:
        - 서비스 품질
        - 음식 품질
        - 위생 및 청결도
        - 가격 및 가치
        - 대기 시간
        - 분위기(내부 인테리어) 및 환경
        - 기타 (위 카테고리에 속하지 않는 문제점)

    3. 2번에서 분류한 각 카테고리별로 다음 정보를 제공하세요:
        a) 해당 카테고리의 문제점이 언급된 리뷰의 수와 비율
        b) 불만 사항들의 구체적이고 자세한 요약 설명 
        c) 각 문제점에 대한 상세한 개선 방안 (최소 3가지 이상)을 우선 순위에 따라 제공
        d) 제시한 개선방안 적용 시 예상되는 효과
        e) 카테고리에 해당하는 불만 사항이 적힌 리뷰 3개 이상

    응답 형식:
    1. 부정적 리뷰 분석 개요
    2. 카테고리별 문제점 분석 및 개선 방안

    각 섹션에 대해 상세하고 구체적인 내용을 제공하세요. 전문적이고 건설적인 톤을 유지하면서, 가게 운영자가 즉시 실행에 옮길 수 있는 실용적인 조언을 제공하세요.
    """

    output = llm_feedback(system_prompt)
    print(output)

def llm_feedback(system_prompt):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "리뷰 데이터를 분석하고 피드백을 제공해주세요."}
        ],
        temperature = 0,
        max_tokens = 2000
    )
    response = completion.choices[0].message.content

    return response

if "__name__" == "__name__":
    main()
