import streamlit as st
import cv2
import numpy as np
import img2pdf
import io

# ================= 설정 및 유틸리티 함수 =================

def adjust_gamma(image, gamma=1.0):
    """이미지의 감마 값을 조절하여 밝기/진하기를 변경합니다."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_readability(img, gamma_value, apply_blur):
    """이미지 화질 개선 함수"""
    # YUV 변환 후 Y 채널(밝기)만 조절
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_y = img_yuv[:,:,0]
    
    # 감마 보정 적용
    img_y = adjust_gamma(img_y, gamma=gamma_value)

    # 노이즈 제거 (선택 사항)
    if apply_blur:
        img_y = cv2.GaussianBlur(img_y, (3, 3), 0.5)

    img_yuv[:,:,0] = img_y
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return result

# ================= Streamlit UI 구성 =================

st.set_page_config(page_title="문서 이미지 화질 개선기", layout="wide")

st.title("porosad님을 위한 화질 개선기")
st.markdown("""
촬영하거나 스캔한 문서 이미지의 글자를 진하게 보정하고, 하나의 PDF로 합쳐줍니다.
""")

# 사이드바 설정
st.sidebar.header("설정")

gamma_val = st.sidebar.slider(
    "글자 진하기 (Gamma)", 
    min_value=0.1, 
    max_value=2.0, 
    value=0.5, 
    step=0.1,
    help="값이 낮을수록 글자가 더 진해집니다. (기본값: 0.5)"
)

use_blur = st.sidebar.checkbox(
    "노이즈 제거 (Blur)", 
    value=True,
    help="글자가 너무 자글거린다면 체크하세요."
)

img_quality = st.sidebar.slider(
    "결과물 JPG 화질",
    min_value=50,
    max_value=100,
    value=85,
    help="PDF 용량을 줄이려면 값을 낮추세요."
)

# 파일 업로드
st.subheader("1. 이미지 업로드")
uploaded_files = st.file_uploader(
    "이미지 파일들을 드래그하거나 선택하세요 (JPG, PNG 등)", 
    type=['png', 'jpg', 'jpeg', 'bmp'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
    
    # 파일명 기준으로 정렬 (업로드 순서가 뒤죽박죽일 수 있으므로 이름순 정렬 권장)
    uploaded_files.sort(key=lambda x: x.name)

    # 미리보기 (첫 번째 이미지)
    with st.expander("첫 번째 이미지 미리보기 (원본 vs 보정본)"):
        # 파일을 바이트로 읽어서 OpenCV 이미지로 변환
        file_bytes = np.asarray(bytearray(uploaded_files[0].read()), dtype=np.uint8)
        preview_img = cv2.imdecode(file_bytes, 1)
        uploaded_files[0].seek(0) # 파일 포인터 초기화

        # 보정 적용
        processed_preview = enhance_readability(preview_img, gamma_val, use_blur)

        col1, col2 = st.columns(2)
        with col1:
            st.image(preview_img, channels="BGR", caption="원본", use_container_width=True)
        with col2:
            st.image(processed_preview, channels="BGR", caption=f"보정본 (Gamma: {gamma_val})", use_container_width=True)

    # 변환 버튼
    if st.button("변환 및 PDF 생성 시작..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_img_bytes_list = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # 진행상황 업데이트
            status_text.text(f"처리 중... ({idx + 1}/{len(uploaded_files)}) : {uploaded_file.name}")
            progress_bar.progress((idx + 1) / len(uploaded_files))

            # 1. 업로드된 파일을 OpenCV 포맷으로 변환
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # 2. 화질 개선 처리
            enhanced_img = enhance_readability(img, gamma_val, use_blur)

            # 3. 이미지를 다시 메모리 상의 JPG 바이트로 인코딩 (디스크 저장 X)
            is_success, buffer = cv2.imencode(".jpg", enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), img_quality])
            
            if is_success:
                processed_img_bytes_list.append(buffer.tobytes())
            else:
                st.error(f"{uploaded_file.name} 변환 실패")

        # 4. PDF 생성
        if processed_img_bytes_list:
            try:
                status_text.text("PDF 병합 중...")
                pdf_bytes = img2pdf.convert(processed_img_bytes_list)
                
                st.success("변환 완료!")
                
                # 다운로드 버튼 생성
                st.download_button(
                    label="PDF 다운로드",
                    data=pdf_bytes,
                    file_name="processed_document.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF 생성 중 오류 발생: {e}")
        else:
            st.warning("처리된 이미지가 없습니다.")