from selenium import webdriver # selenium : 동적으로 웹 사이트를 움직이는 역할
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

def main():
    query = input("네이버에 등록된 가게 이름을 입력해주세요! : ")
    # query = "자주오리돌판구이 용인본점" # 검색어 설정 # 네이버에 등록된 이름으로 
    url = f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}"

    response = requests.get(url)# 웹 페이지 요청
    html = response.text
    soup = BeautifulSoup(html, 'html.parser') # BeautifulSoup으로 HTML 파싱

    div_tag = soup.find('div', class_='LylZZ') # div 태그의 class가 'LylZZ'인 요소 찾기
    a_tag = div_tag.find('a')['href']# 해당 div 안에 있는 a 태그의 href 속성 추출
    match = re.search(r'place/(\d+)', a_tag) # 정규 표현식을 사용하여 place/ 뒤의 숫자만 추출 (가게 ID)
    store_id = match.group(1)  # 첫 번째 그룹 (숫자)

    # 정보 크롤링 해오기
    tab_list = ["information", "menu/list", "feed", "home", "review/visitor"] # 정보, 메뉴, 소식, 홈, 리뷰
    class_name_list = ["T8RFa", "place_section_content", "place_section_content", "PIbes", "pui__vn15t2"]
    result ={}

    for tab, class_name in zip(tab_list, class_name_list):
        url = f"https://pcmap.place.naver.com/restaurant/{store_id}/{tab}"

        driver = webdriver.Chrome() # iframe 없는 url 로드 
        driver.get(url)

        if (tab == "review/visitor"): # 리뷰일 때
            before_h = driver.execute_script("return window.scrollY") # 스크롤 전 높이
            cnt = 0
            while (cnt<10): # 무한 스크롤 (지금은 10번만) # 110개 
                try:
                    # '더보기' 버튼이 있는지 확인하고 클릭합니다.
                    load_more_button = driver.find_element(By.XPATH, "//a[@class='fvwqf']")
                    load_more_button.click()  # 더보기 버튼 클릭
                    time.sleep(1)  # 새로운 리뷰가 로드될 시간을 대기
                except:
                    # '더보기' 버튼이 없으면 그냥 스크롤만 진행합니다.
                    pass

                driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END) # 맨 아래로 스크롤 내린다.
                time.sleep(1) # 스크롤 사이 페이지 로딩 시간

                # 스크롤 후 높이
                after_h = driver.execute_script("return window.scrollY")
                if after_h == before_h:
                    break
                before_h = after_h
                cnt += 1

            content =  driver.find_elements(By.CLASS_NAME, class_name) # 스크롤 끝나면 정보 가져오기
            content = [element.text for element in content]
        else:
            content =  driver.find_element(By.CLASS_NAME, class_name).text # 정보 가져오기

        result[tab] = content

        driver.quit()

    results = pd.DataFrame({key: [value] for key, value in result.items() if key != 'review/visitor'}) # 'review/visitor'를 제외한 컬럼들로 DataFrame 생성
    reviews = pd.DataFrame(result['review/visitor'], columns=['review']) # 'review/visitor' 데이터로 별도의 DataFrame 생성

    results.to_csv(r"gpt_contest\info.csv")
    reviews.to_csv(r"gpt_contest\reviews.csv")

if "__name__" == "__name__":
    main()