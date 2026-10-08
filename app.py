import streamlit as st
import cv2
import numpy as np
import os
import io
import base64
import requests

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

st.title("📸 증명사진 수집 및 자동 스캔")
st.write("학번과 이름을 입력하고 증명사진을 업로드하면, AI가 자동으로 수평을 맞추고 보정하여 선생님 클라우드에 안전하게 저장됩니다.")

# 학생 입력 폼
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("학번 (예: 10501)")
with col2:
    student_name = st.text_input("이름 (예: 홍길동)")

uploaded_file = st.file_uploader("증명사진 파일 업로드 (스마트폰 촬영 또는 파일 선택)", type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

def order_points(pts):
    """4개의 꼭짓점을 시계 방향(좌상, 우상, 우하, 좌하)으로 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def process_single_photo(image):
    """선생님이 검증하신 로컬 스캔 알고리즘을 단일 사진에 적용"""
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # [핵심 1] 빛 번짐이나 스캔/촬영 노이즈에 강한 적응형 이진화 + 엣지 추출 혼합
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 150)
    
    combined_mask = cv2.bitwise_or(thresh_adapt, edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_area = 0
    img_area = img_h * img_w

    for c in contours:
        # [핵심 2] Convex Hull 적용: 삐뚤빼뚤한 테두리를 빳빳하게 폄
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        
        w, h = rect[1]
        if w == 0 or h == 0: continue
        if w > h: w, h = h, w
        area = w * h
        
        # [핵심 3] 엄격한 증명사진 비율 검증 (0.65 ~ 0.90)
        if img_area * 0.005 < area < img_area * 0.95:
            if 0.65 < (w / h) < 0.90:
                if area > max_area:
                    max_area = area
                    box = cv2.boxPoints(rect)
                    best_box = np.intp(box)
                    
    if best_box is not None:
        ordered_pts = order_points(best_box)
        target_w, target_h = 350, 450
        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        # 사진을 못 찾은 경우 원본을 그대로 쓰지 않고 중앙 크롭 형태로 안전하게 대응
        warped = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)

# [수정] 검은 테두리와 바깥 잔상을 확실히 날려버리도록 마진을 살짝 늘림
    trim_x = 12       # 좌우 폭을 조금 더 안쪽으로 잘라냄
    trim_y_top = 4    # 정수리는 안전하게 보호하면서 미세 테두리 제거
    trim_y_bottom = 12  # 아래쪽 여백 및 그림자 제거
    
    final_crop = warped[trim_y_top:target_h-trim_y_bottom, trim_x:target_w-trim_x]
    final_image = cv2.resize(final_crop, (350, 450), interpolation=cv2.INTER_CUBIC)
    return final_image


def upload_via_gas(file_bytes, file_name):
    """구글 앱스크립트 웹앱을 통해 개인 드라이브로 안전하게 전송"""
    gas_url = st.secrets["gas_url"]
    
    encoded_data = base64.b64encode(file_bytes).decode('utf-8')
    payload = {
        "fileData": encoded_data,
        "fileName": file_name,
        "mimeType": "image/jpeg"
    }
    
    response = requests.post(gas_url, json=payload)
    result = response.json()
    if result.get("status") != "success":
        raise Exception(result.get("message", "알 수 없는 전송 오류"))

# 제출 버튼
if st.button("사진 제출 및 자동 스캔 완료", type="primary"):
    if not student_id or not student_name:
        st.warning("⚠️ 학번과 이름을 모두 입력해주세요!")
    elif not uploaded_file:
        st.warning("⚠️ 증명사진 파일을 업로드해주세요!")
    else:
        with st.spinner("사진을 분석하고 정밀 스캔하여 구글 드라이브에 안전하게 저장 중입니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                processed_img = process_single_photo(image)
                
                is_success, buffer = cv2.imencode(".jpg", processed_img)
                if not is_success:
                    st.error("이미지 변환 중 오류가 발생했습니다.")
                else:
                    file_name = f"{student_id}_{student_name}.jpg"
                    upload_via_gas(buffer.tobytes(), file_name)
                    
                    st.success(f"🎉 [{student_id} {student_name}] 학생의 증명사진이 성공적으로 제출 및 자동 보정되었습니다!")
                    st.image(processed_img, channels="BGR", caption="선생님 드라이브에 저장된 최종 보정 사진")
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")
