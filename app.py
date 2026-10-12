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
if st.session_state.get("is_teacher_creator", False) and current_code:
    st.success("🎉 **우리 반 전용 링크가 활성화되었습니다!**")
    st.info("👉 **현재 브라우저 주소창의 전체 주소를 복사하여 학생들에게 공유**해 주세요.\n(학생들이 접속할 때는 이 안내 배너와 설정창이 자동으로 숨겨집니다.)")

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

def order_points_quad(pts):
    """4개의 꼭짓점을 시계 방향(좌상, 우상, 우하, 좌하)으로 정렬하고 증명사진 세로 방향 보정"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    center = np.mean(pts, axis=0)
    
    # 중심점 기준 각도 정렬 (시계 방향 정렬)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts_sorted = pts[np.argsort(angles)]
    
    # 좌상단(Top-Left: x+y 최소) 점을 0번 인덱스로 회전
    tl_idx = np.argmin(pts_sorted.sum(axis=1))
    ordered = np.roll(pts_sorted, -tl_idx, axis=0)
    
    # 가로/세로 길이 판별
    w = (np.linalg.norm(ordered[0] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[2])) / 2.0
    h = (np.linalg.norm(ordered[1] - ordered[2]) + np.linalg.norm(ordered[0] - ordered[3])) / 2.0
    
    # 가로로 누워있는 경우 시계 방향 90도 회전
    if w > h:
        ordered = np.roll(ordered, -1, axis=0)
        
    return ordered.astype("float32")

def process_single_photo(image):
    """얼굴만 파고드는 과도한 크롭을 막고, 인화지 전체를 온전히 스캔하는 보정 엔진"""
    img_h, img_w = image.shape[:2]
    
    # 1. 순수 디지털 원본 파일인지 판별 (가장자리 평균 밝기)
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
    
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 150)
    combined_mask = cv2.bitwise_or(thresh_adapt, edged)
    
    # 인화지 외곽선을 끊김 없이 연결하기 위한 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = img_h * img_w
    candidates = []

    for c in contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
        
        # 4각형 후보 우선 추출
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            area = cv2.contourArea(approx)
        else:
            rect = cv2.minAreaRect(hull)
            area = rect[1][0] * rect[1][1]
            pts = cv2.boxPoints(rect)
            
        # 스마트폰 촬영 환경에서 전형적인 증명사진 크기 (0.5% ~ 15%)
        if not (img_area * 0.005 < area < img_area * 0.15):
            continue
            
        rect_eval = cv2.minAreaRect(np.float32(pts))
        w, h = rect_eval[1]
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        aspect_ratio = w / h
        
        # 증명사진 규격 비율 검증 (0.68 ~ 0.88)
        if not (0.68 < aspect_ratio < 0.88):
            continue
            
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [np.int32(pts)], -1, 255, -1)
        mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
        mean_val = float(mean_val[0][0])
        std_val = float(std_val[0][0])
        
        # 텅 빈 플라스틱 바닥(std < 20) 및 어두운 소품(mean < 70) 배제
        if std_val < 20 or mean_val < 70:
            continue
            
        # 임시 워핑 검사로 폴라로이드 감지
        ordered_test = order_points_quad(pts)
        dst_test = np.array([[0, 0], [99, 0], [99, 129], [0, 129]], dtype="float32")
        M_test = cv2.getPerspectiveTransform(ordered_test, dst_test)
        temp_crop = cv2.warpPerspective(gray, M_test, (100, 130))
        
        # 폴라로이드 판별 (하단 20% 영역이 두꺼운 흰색 여백인지 검사)
        bottom_strip = temp_crop[105:, 15:85]
        if np.mean(bottom_strip) > 175 and np.std(bottom_strip) < 22:
            continue  # 폴라로이드 사진 탈락
            
        # 비율 적합도 + 면적 가중치 (얼굴 내부 조각 대신 인화지 전체 외곽 우선)
        ratio_diff = min(abs(aspect_ratio - 0.75), abs(aspect_ratio - 0.778))
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.10)) * 100.0
        area_score = (area / img_area) * 500.0  # 더 큰 외곽 인화지 우대
        is_quad_bonus = 30.0 if len(approx) == 4 else 0.0
        
        total_score = ratio_score + area_score + is_quad_bonus
        candidates.append((total_score, pts))
        
    best_box = None
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_box = candidates[0][1]

    if best_box is not None:
        ordered_pts = order_points_quad(best_box)
        
        target_w, target_h = 350, 450
        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        warped = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)

    # 어깨와 머리가 잘리지 않도록 미세 테두리만 정리 (2픽셀 안전 마진)
    trim_x = 2
    trim_y = 2
    final_crop = warped[trim_y:target_h-trim_y, trim_x:target_w-trim_x]
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
