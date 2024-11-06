import streamlit as st
import tiktoken
from loguru import logger  # 파일 업로드 및 처리 상황을 로그로 기록, 디버깅과 기록용

from langchain.chains import ConversationalRetrievalChain # 대화형 검색 체인을 구성
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter  # 청크 
from langchain.embeddings import HuggingFaceEmbeddings # 허깅페이스 임베딩 모델

from langchain.memory import ConversationBufferMemory # 대화내용을 메모리에 저장
from langchain.vectorstores import FAISS # 고속 벡터 검색 라이브러리, 임베딩 저장, 빠르게 검색 가능

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config( 
    page_title="OneClickMakerChatbot",  # 웹페이지 타이틀
    page_icon="💬")   # 웹페이지 아이콘

    st.title("_Chatbot :blue[입니다!]_ 💩")  # 제목(언더바 == 기울임꼴)

    # conversation 키가 session_state에 없으면 새로운 None 값을 할당하여 초기화
    if "conversation" not in st.session_state: # 대화 체인의 상태를 저장
        st.session_state.conversation = None

    if "chat_history" not in st.session_state: # 대화 내역 관리
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state: # 파일 처리 상태 저장
        st.session_state.processComplete = None


    with st.sidebar: # 왼쪽 사이드바 구성
        st.header("업종 선택")
        business_type = st.selectbox( # 업종 선택 메뉴
        "업종을 선택하세요",
        ["음식점", "미용실", "쇼핑몰", "부동산", "관광숙박업"]
        )

        st.header("PDF 제출")
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf','docx'],accept_multiple_files=True)
        process = st.button("Process")  # 파일을 업로드 후 제출 버튼
        
    # API 키 설정
    openai_api_key = "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"

    if process: # process 버튼이 눌렸을 때 실행
        files_text = get_text(uploaded_files)  # 업로드된 파일을 텍스트로 변환.
        text_chunks = get_text_chunks(files_text) # 텍스트를 청크로 나눔
        vetorestore = get_vectorstore(text_chunks) # 벡터화하여 저장

        # LLM이 이 벡터store를 갖고 답변할 수 있도록 체인 구성
        # 이 체인은 벡터화된 텍스트 저장소와 OpenAI API 키를 사용하여 대화를 수행할 수 있게 합
        # get함수로 대화 체인을 생성, session_state의 conversaton키에 할당
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 
        st.session_state.processComplete = True # 파일 처리 완료


    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",  # 환영 메세지
                                        "content": "안녕하세요! 궁금한 것이 있으면 언제든 물어봐주세요!"}] 
        # messages 키가 session_state에 없을 경우 초기화하고, 기본 환영 메시지를 설정
        # role: 발신자 / content: 메세지 내용 저장
    for message in st.session_state.messages: # 대화 내용을 반복해서 표시
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # 아이콘, 대화

    history = StreamlitChatMessageHistory(key="chat_messages")  # 메모리 구현

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):  # 질문창
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):  # 사용자가 질문을 입력하면
            st.markdown(query)

        with st.chat_message("assistant"): # 대화 세션에 추가하고,
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): # 로딩 기호
                result = chain({"question": query})  # 모델이 생성한 답변을 response에 저장 후 표시.
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']  # 참고한 문서 확인

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    #st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    #st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    
# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):  # 토큰 길이 계산: 텍스트를 토큰으로 변환해 길이를 계산.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):  # 업로드된 파일을 텍스트로 변환

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):  # 텍스트를 청크로 나누어 줌. (chunk_size: 청크 크기, chunk_overlap: 중첩 길이)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):  # 텍스트 청크를 벡터로 변환해 FAISS 벡터 스토어에 저장.
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key): #대화 체인 생성: LLM이 벡터 스토어와 상호작용 
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain


if __name__ == '__main__':
    main()




