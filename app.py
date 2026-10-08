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

# --- ⚙️ 선생님 설정 패널 (쌤동네 배포용 관리자 모드) ---
with st.expander("⚙️ [선생님 전용 설정] 내 구글 드라이브 연결 (처음 접속 시 1회 입력)"):
    st.markdown("다른 학교 선생님이신가요? 본인의 **구글 앱스크립트(GAS) 웹 앱 URL**을 아래에 입력하고 저장하면, 학생들이 올린 사진이 선생님의 개인 구글 드라이브로 바로 전송됩니다.")
    
    if "gas_url" not in st.session_state:
        st.session_state["gas_url"] = ""
        
    input_url = st.text_input("구글 앱스크립트(GAS) 웹 앱 URL 입력", value=st.session_state["gas_url"], type="password")
    if st.button("설정 저장하기"):
        st.session_state["gas_url"] = input_url.strip()
        st.success("✨ 설정이 저장되었습니다! 이제 학생들이 이 링크로 사진을 제출할 수 있습니다.")

st.markdown("---")

# 학생 입력 폼
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("학번 (예: 1학년 1반 1번인 경우, 10101)")
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
    """실물 촬영 사진과 순수 디지털 증명사진 파일을 스마트하게 구분하여 보정"""
    img_h, img_w = image.shape[:2]
    
    # [스마트 구분 로직] 이미지의 4방향 가장자리(테두리 5픽셀) 평균 밝기 측정
    # - 책상/마우스패드 위에서 찍은 실물 사진: 테두리가 어두움 (낮은 밝기)
    # - 순수 디지털 증명사진 파일: 테두리가 하얗거나 밝음 (높은 밝기)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    avg_brightness = np.mean(border_concat)
    
    # 만약 테두리가 밝고 하얀 경우 (이미 완성된 디지털 증명사진 파일인 경우)
    if avg_brightness > 140:
        # 불필요한 사선 왜곡 없이 곧바로 표준 규격(350x450)으로 깔끔하게 리사이즈
        return cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
    
    # ==========================================
    # 아래는 테두리가 어두운 [실물 촬영 사진]인 경우에만 작동하는 정밀 스캔 로직
    # ==========================================
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
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if img_area * 0.005 < area < img_area * 0.95:
                rect = cv2.minAreaRect(approx)
                w, h = rect[1]
                if w > h: w, h = h, w
                if h > 0 and 0.65 < (w / h) < 0.90:
                    if area > max_area:
                        max_area = area
                        best_box = approx.reshape(4, 2)
                        
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
        center = np.mean(ordered_pts, axis=0)
        ordered_pts = center + (ordered_pts - center) * 0.985
        
        target_w, target_h = 350, 450
        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        warped = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)

    trim_x = 8
    trim_y_top = 2     
    trim_y_bottom = 8  
    
    final_crop = warped[trim_y_top:target_h-trim_y_bottom, trim_x:target_w-trim_x]
    final_image = cv2.resize(final_crop, (350, 450), interpolation=cv2.INTER_CUBIC)
    return final_image

def upload_via_gas(file_bytes, file_name):
    """구글 앱스크립트 웹앱을 통해 개인 드라이브로 안전하게 전송"""
    gas_url = st.session_state.get("gas_url", "").strip()
    if not gas_url:
        raise Exception("선생님 설정에서 구글 앱스크립트(GAS) URL이 입력되지 않았습니다! 상단의 [선생님 전용 설정]을 열어 URL을 먼저 등록해주세요.")
    
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
    if not st.session_state.get("gas_url"):
        st.error("🚨 상단의 **[선생님 전용 설정]**을 펼쳐서 본인의 구글 앱스크립트(GAS) URL을 먼저 등록해주세요!")
    elif not student_id or not student_name:
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
