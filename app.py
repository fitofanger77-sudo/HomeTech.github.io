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
# 선생님이 링크를 방금 발급한 순간에만 안내 배너 표시 (학생에게는 숨김)
if st.session_state.get("is_teacher_creator", False) and current_code:
    st.success("🎉 **우리 반 전용 링크가 활성화되었습니다!**")
    st.info("👉 **현재 브라우저 주소창의 전체 주소를 복사하여 학생들에게 공유**해 주세요.\n(학생들이 접속할 때는 이 안내 배너와 설정창이 자동으로 숨겨집니다.)")

# 전용 코드가 없는 기본 주소로 접속했을 때만 선생님 설정창 표시
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
    """복잡한 배경 및 잡화 속에서도 진짜 증명사진만 콕 집어내는 고정밀 스캔 알고리즘"""
    img_h, img_w = image.shape[:2]
    
    # 1. 디지털 원본 파일인지 판별 (가장자리 평균 밝기)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    avg_brightness = np.mean(border_concat)
    
    if avg_brightness > 140:
        return cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
    
    # 2. 실물 촬영 사진 정밀 분석
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1) 조명 변화에 적응하는 오츠(Otsu) 이진화 (흰 인화지 분리)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2) 고정 임계값 이진화 (어두운/회색 배경에서 흰 종이 분리)
    _, thresh_bright = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    
    # 3) 적응형 이진화 + Canny (나무 책상 등 경계선 추출)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 150)
    combined_mask = cv2.bitwise_or(thresh_adapt, edged)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_combined = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours_otsu, _ = cv2.findContours(closed_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_bright, _ = cv2.findContours(closed_bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_combined, _ = cv2.findContours(closed_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours = list(contours_otsu) + list(contours_bright) + list(contours_combined)
    img_area = img_h * img_w
    candidates = []

    for c in all_contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        is_quad = False
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts_to_check = approx.reshape(4, 2)
            area = cv2.contourArea(approx)
            is_quad = True
        else:
            rect = cv2.minAreaRect(hull)
            area = rect[1][0] * rect[1][1]
            pts_to_check = cv2.boxPoints(rect)
            
        # 너무 작거나(0.3% 미만) 화면 전체를 덮는 배경(90% 초과) 제거
        if not (img_area * 0.003 < area < img_area * 0.90):
            continue
            
        rect = cv2.minAreaRect(np.float32(pts_to_check))
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        aspect_ratio = w / h
        
        # 증명사진 종횡비 (0.65 ~ 0.90) - 한국 표준 3.5:4.5 = 0.778
        if not (0.65 < aspect_ratio < 0.90):
            continue
            
        # 후보 영역 마스크 생성 후 밝기 및 질감(표준편차) 검증
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [np.int32(pts_to_check)], -1, 255, -1)
        mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
        mean_val = float(mean_val[0][0])
        std_val = float(std_val[0][0])
        
        # [핵심 검증 1] 텅 빈 서랍 플라스틱 바닥, 마우스패드 차단 (표준편차 22 미만 탈락)
        if std_val < 22:
            continue
            
        # [핵심 검증 2] 어두운 자물쇠나 잡동사니 차단 (평균 밝기 70 미만 탈락)
        if mean_val < 70:
            continue
            
        # 적합도 채점 (비율 일치도 + 인물 질감 + 밝기 + 사각형 보너스)
        ratio_diff = abs(aspect_ratio - 0.778)
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.15)) * 100.0
        texture_score = min(std_val / 50.0, 1.5) * 50.0
        bright_score = min(mean_val / 180.0, 1.0) * 40.0
        shape_bonus = 30.0 if is_quad else 0.0
        
        total_score = ratio_score + texture_score + bright_score + shape_bonus
        candidates.append((total_score, pts_to_check))
        
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
