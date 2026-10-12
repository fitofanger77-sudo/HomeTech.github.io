import streamlit as st
import cv2
import numpy as np
import os
import io
import base64
import requests

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

# 1. URL 주소에서 선생님 고유 코드(?t=...) 자동 인식 (Secrets 백업 호환)
current_code = st.query_params.get("t", "")
target_gas_url = ""

if current_code:
    try:
        target_gas_url = base64.urlsafe_b64decode(current_code.encode("utf-8")).decode("utf-8")
    except Exception:
        target_gas_url = ""
elif "gas_url" in st.secrets:
    target_gas_url = st.secrets["gas_url"]

st.title("📸 증명사진 수집 및 자동 스캔")
st.write("학번과 이름을 입력하고 증명사진을 업로드하면, AI가 자동으로 수평을 맞추고 보정하여 선생님 클라우드에 안전하게 저장됩니다.")

# --- ⚙️ 선생님 전용 영역 제어 ---
# (1) 선생님이 링크를 방금 발급한 순간에만 안내 배너 노출 (학생에게는 보이지 않음)
if st.session_state.get("is_teacher_creator", False) and current_code:
    st.success("🎉 **우리 반 전용 링크가 활성화되었습니다!**")
    st.info("👉 **현재 브라우저 주소창의 전체 주소를 복사하여 학생들에게 공유**해 주세요.\n(학생들이 접속할 때는 이 안내 배너와 설정창이 자동으로 숨겨집니다.)")

# (2) 전용 코드가 없는 기본 주소로 접속했을 때만 선생님 설정창 표시
if not current_code:
    with st.expander("⚙️ [선생님 전용] 우리 반 전용 수집 링크 만들기", expanded=True):
        st.markdown("구매하신 선생님이신가요? 본인의 **구글 앱스크립트(GAS) 웹 앱 URL**을 입력하시면, **우리 반 학생들에게 배포할 전용 링크**를 즉시 만들어 드립니다.")
        gas_input = st.text_input("구글 앱스크립트(GAS) 웹 앱 URL 입력", value=target_gas_url, type="password")
        
        if st.button("우리 반 전용 링크 발급하기"):
            if not gas_input.strip():
                st.warning("⚠️ 구글 앱스크립트 URL을 입력해주세요!")
            else:
                encoded_code = base64.urlsafe_b64encode(gas_input.strip().encode("utf-8")).decode("utf-8")
                st.session_state["is_teacher_creator"] = True
                st.query_params["t"] = encoded_code
                st.rerun()

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
    
    # [스마트 구분] 테두리 4변의 평균 밝기 측정 (디지털 파일 감지)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    avg_brightness = np.mean(border_concat)
    
    # 이미 완성된 디지털 파일인 경우 왜곡 없이 리사이즈
    if avg_brightness > 140:
        return cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
    
    # ==========================================
    # 실물 촬영본 정밀 감지 로직 (어두운 배경 대응)
    # ==========================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1) 흰 종이 추출 전용 밝기 마스크 (마우스패드 등 어두운 바닥 분리)
    _, thresh_bright = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    
    # 2) 적응형 이진화 + Canny 엣지 (일반 바닥/나무 책상 분리)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 150)
    combined_mask = cv2.bitwise_or(thresh_adapt, edged)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_combined = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours_bright, _ = cv2.findContours(closed_bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_combined, _ = cv2.findContours(closed_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours = list(contours_bright) + list(contours_combined)
    img_area = img_h * img_w
    candidates = []

    for c in all_contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        pts_to_check = None
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts_to_check = approx.reshape(4, 2)
            area = cv2.contourArea(approx)
        else:
            rect = cv2.minAreaRect(hull)
            area = rect[1][0] * rect[1][1]
            pts_to_check = cv2.boxPoints(rect)
            
        if not (img_area * 0.005 < area < img_area * 0.95):
            continue
            
        rect = cv2.minAreaRect(np.float32(pts_to_check))
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        aspect_ratio = w / h
        
        # 증명사진 비율 (0.60 ~ 0.95)
        if not (0.60 < aspect_ratio < 0.95):
            continue
            
        # [핵심 검증] 후보 영역 내부의 평균 밝기 확인
        # 검은 마우스패드(밝기 20~50)는 탈락, 흰 증명사진 종이(밝기 80 이상)만 선별
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [np.int32(pts_to_check)], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        
        if mean_val < 80:
            continue
            
        # 규격 적합도 채점 (3.5x4.5 비율 기준 + 밝기 + 다각형 가산점)
        ratio_diff = abs(aspect_ratio - 0.77)
        shape_bonus = 20 if (len(approx) == 4 and cv2.isContourConvex(approx)) else 0
        score = (1.0 - ratio_diff) * 100 + (mean_val / 255.0) * 30 + shape_bonus
        candidates.append((score, pts_to_check))
        
    best_box = None
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_box = candidates[0][1]

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

def upload_via_gas(file_bytes, file_name, gas_url):
    """구글 앱스크립트 웹앱을 통해 개인 드라이브로 안전하게 전송"""
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
    if not target_gas_url:
        st.error("🚨 담임 선생님의 구글 드라이브 연결 정보가 없습니다! 담임 선생님께서 공유해주신 전용 링크로 다시 접속해주세요.")
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
                    upload_via_gas(buffer.tobytes(), file_name, target_gas_url)
                    
                    st.success(f"🎉 [{student_id} {student_name}] 학생의 증명사진이 성공적으로 제출 및 자동 보정되었습니다!")
                    st.image(processed_img, channels="BGR", caption="선생님 드라이브에 저장된 최종 보정 사진")
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")

# 학생 화면 하단 관리자 복귀 링크
if current_code and not st.session_state.get("is_teacher_creator", False):
    st.markdown("<br><hr><center><small style='color: gray;'>선생님이신가요? <a href='.' style='color: gray; text-decoration: underline;'>새 링크 만들기</a></small></center>", unsafe_allow_html=True)
