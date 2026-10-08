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
    """사선 꼭짓점 정밀 탐지 및 안쪽 수축 마진을 적용한 스캔 보정 알고리즘"""
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
        hull = cv2.convexHull(c)
        # 사선 꼭짓점을 더 정밀하게 찾기 위해 다각형 근사화(approxPolyDP) 시도
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        # 꼭짓점이 4개이면서 일정 크기 이상인 경우 우선 채택
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if img_area * 0.005 < area < img_area * 0.95:
                # 가로세로 비율 검증 (증명사진 비율 0.65 ~ 0.90)
                rect = cv2.minAreaRect(approx)
                w, h = rect[1]
                if w > h: w, h = h, w
                if h > 0 and 0.65 < (w / h) < 0.90:
                    if area > max_area:
                        max_area = area
                        best_box = approx.reshape(4, 2)
                        
    # 만약 4꼭짓점 다각형이 깔끔하게 안 잡혔다면 기존 minAreaRect 박스로 안전하게 백업
    if best_box is None:
        for c in contours:
            hull = cv2.convexHull(c)
            rect = cv2.minAreaRect(hull)
            w, h = rect[1]
            if w == 0 or h == 0: continue
            if w > h: w, h = h, w
            area = w * h
            if img_area * 0.005 < area < img_area * 0.95 and 0.65 < (w / h) < 0.90:
                if area > max_area:
                    max_area = area
                    best_box = cv2.boxPoints(rect)
                    
    if best_box is not None:
        ordered_pts = order_points(np.array(best_box, dtype="float32"))
        
        # [핵심 아이디어] 찾은 꼭짓점들을 중심점(Centroid) 기준 안쪽으로 1% 정도 살짝 당겨서 테두리 검은 선 원천 차단
        center = np.mean(ordered_pts, axis=0)
        ordered_pts = center + (ordered_pts - center) * 0.985
        
        target_w, target_h = 350, 450
        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        warped = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)

    # 최종 마진 컷 (정수리는 보호하고 미세한 바깥 잔상 제거)
    trim_x = 8
    trim_y_top = 2     
    trim_y_bottom = 8  
    
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
