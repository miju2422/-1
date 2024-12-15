#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

st.title("2422 서비스제작 수행")

st.subheader("5개의 애니를 분류해보자")

# Google Drive 파일 ID
file_id = '19kObuLXVejEivq75gZPGGNPvkHsrS33A'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/QcvCJPZ/3.jpg?text=Label1_Image1",
            "https://i.ibb.co/GTgtDP8/4.jpg?text=Label1_Image2",
            "https://i.ibb.co/f2BtvLv/1.jpg?text=Label1_Image3"
        ],
        'texts': [
            "애니의 제목은 슬램덩크에요.",
            "농구동아리에 관련된 이야기를 다루고 있어요.",
            "제작자는 이 애니를 보지 안았어."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/QcvCJPZ/3.jpg?text=Label1_Image1",
            "https://i.ibb.co/GTgtDP8/4.jpg?text=Label1_Image2",
            "https://i.ibb.co/f2BtvLv/1.jpg?text=Label1_Image3"
        ],
        'texts': [
            "애니의 제목은 윈브레에요.",
            "학교를 배경으로 한 싸움을 다루고 있어요. 싸움신을 잘 만든 애니 중 하니에요",
            "매력적인 캐릭터들이 정말 많아요. 보고있으면 눈이 행복해요~"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/QcvCJPZ/3.jpg?text=Label1_Image1",
            "https://i.ibb.co/GTgtDP8/4.jpg?text=Label1_Image2",
            "https://i.ibb.co/f2BtvLv/1.jpg?text=Label1_Image3"
        ],
        'texts': [
            "애니의 제목은 츠루네=카제마이고교 궁도부-에요.",
            "이 애니는 궁도동아리에 대한 이야기를 다루고 있어요.",
            "주인공이 슬럼프에 빠져 궁도를 그만두게 되요. 하지만 고등학교에 진학하면서 다시 궁도를 시작하고 결국 슬럼프를 극복한다는 이야기에요."
        ]
    },
    labels[3]: {
        'images': [
            "https://i.ibb.co/QcvCJPZ/3.jpg?text=Label1_Image1",
            "https://i.ibb.co/GTgtDP8/4.jpg?text=Label1_Image2",
            "https://i.ibb.co/f2BtvLv/1.jpg?text=Label1_Image3"
        ],        
        'texts': [
            "애니의 제목은 치하야후루에요.",
            "이 애니는 일본의 전통게임인 카루타와 관련된 동아리에 대한 이야기를 다루고 있어요.",
            "초등학생 시절의 어떤 사건으로 카루타에 흥미를 갖게 된 여주인공이 카루타로 최고에 자리에 도달하기 위해 노력하는 내용을 담고 있어."
        ]
    },
    labels[4]: {
        'images': [
            "https://i.ibb.co/QcvCJPZ/3.jpg?text=Label1_Image1",
            "https://i.ibb.co/GTgtDP8/4.jpg?text=Label1_Image2",
            "https://i.ibb.co/f2BtvLv/1.jpg?text=Label1_Image3"
        ],        
        'texts': [
            "애니의 제목은 하이큐에요.",
            "이 애니는 배구동아리에 대한 이야기를 다루고 있어요.",
            "신장이 작은 주인공이 배구를 동경하게 되면서 시작되는 이야기에요."
        ]
    },
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

