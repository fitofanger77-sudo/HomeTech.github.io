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

def robust_order_points(pts):
    """어떤 각도로 기울어져 있어도 좌표 중복 없이 좌상, 우상, 우하, 좌하 순서로 완벽 정렬"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    ordered = np.array([tl, tr, br, bl], dtype="float32")
    
    # 가로/세로 판별 후 세로 방향으로 정렬 유지
    w = (np.linalg.norm(ordered[0] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[2])) / 2.0
    h = (np.linalg.norm(ordered[1] - ordered[2]) + np.linalg.norm(ordered[0] - ordered[3])) / 2.0
    if w > h:
        ordered = np.roll(ordered, -1, axis=0)
        
    return ordered

def crop_box(img, pts):
    """4개의 꼭짓점 좌표를 기반으로 원근 보정 및 리사이즈"""
    ordered_pts = robust_order_points(pts)
    center = np.mean(ordered_pts, axis=0)
    ordered_pts = center + (ordered_pts - center) * 0.985
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    
    final_crop = warped[2:target_h-2, 2:target_w-2]
    return cv2.resize(final_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def is_actual_portrait_photo(cropped_img):
    """자물쇠, 플라스틱 등 잡동사니를 걸러내고 실제 인물 사진인지 정밀 판별"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # 얼굴이 위치하는 중심 영역 (상단 15%~75%, 좌우 20%~80%)
    roi = hsv[int(h * 0.15):int(h * 0.75), int(w * 0.2):int(w * 0.8)]
    
    # 피부톤 HSV 범위 검출
    lower_skin1 = np.array([0, 25, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 190, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 25, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 190, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(roi, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(roi, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    skin_ratio = np.sum(skin_mask > 0) / (roi.shape[0] * roi.shape[1])
    
    # 플라스틱(1%) 및 자물쇠(3%) 같은 사물은 탈락 (피부색 비율 10% 이상만 통과)
    return skin_ratio >= 0.10

def extract_all_photos(image):
    """잡동사니를 배제하고 오직 실제 인물 사진만 추출"""
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    # 1. 디지털 파일 여부 검사 (가장자리 평균 밝기)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    if np.mean(border_concat) > 140:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_bright = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 150)
    combined = cv2.bitwise_or(thresh_adapt, edged)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_comb = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    c1, _ = cv2.findContours(closed_comb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(closed_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c3, _ = cv2.findContours(closed_bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts = list(c1) + list(c2) + list(c3)
    
    candidates = []
    for c in all_cnts:
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        area = w * h
        
        # 비율 및 크기 필터
        if not (0.62 < ar < 0.95):
            continue
        if not (img_area * 0.008 < area < img_area * 0.35):
            continue
            
        pts = cv2.boxPoints(rect)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [np.int32(pts)], -1, 255, -1)
        mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
        mean_val = float(mean_val[0][0])
        std_val = float(std_val[0][0])
        
        if std_val < 25 or mean_val < 75:
            continue
            
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
        is_quad = 1.0 if (len(approx) == 4 and cv2.isContourConvex(approx)) else 0.0
        
        ratio_diff = min(abs(ar - 0.75), abs(ar - 0.778))
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.15)) * 100.0
        score = (is_quad * 40.0) + ratio_score + std_val + (mean_val / 2.0)
        
        candidates.append({
            'pts': pts,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area,
            'score': score
        })
        
    if not candidates:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    candidates.sort(key=lambda x: (x['score'], x['area']), reverse=True)
    
    # NMS (중복 제거)
    distinct = []
    for cand in candidates:
        cx1, cy1 = cand['center']
        is_overlap = False
        for kept in distinct:
            cx2, cy2 = kept['center']
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < min(cand['w'], kept['w']) * 0.7:
                is_overlap = True
                break
        if not is_overlap:
            if cand['score'] > 120:
                distinct.append(cand)
                
    if not distinct:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    # [핵심 검증] 크롭된 이미지 중 실제 인물 피부가 존재하는 사진만 최종 채택
    verified_photos = []
    # 정렬: 위에서 아래, 좌에서 우
    distinct.sort(key=lambda x: (x['center'][1] // 200, x['center'][0]))
    
    for p in distinct:
        crop_img = crop_box(image, p['pts'])
        if is_actual_portrait_photo(crop_img):
            verified_photos.append(crop_img)
            
    # 만약 피부 검증으로 모두 걸러졌다면 가장 유력한 1개만 안전 반환
    if not verified_photos:
        return [crop_box(image, distinct[0]['pts'])]
        
    return verified_photos

def upload_via_gas(file_bytes, file_name, gas_url):
    """구글 앱스크립트 웹앱을 통해 개인 드라이브로 전송"""
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
        with st.spinner("사진을 분석하고 자동 분할 스캔하여 구글 드라이브에 안전하게 저장 중입니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # 다중 사진 추출 및 인물 검증
                processed_photos = extract_all_photos(image)
                uploaded_names = []
                
                for idx, photo in enumerate(processed_photos):
                    is_success, buffer = cv2.imencode(".jpg", photo)
                    if not is_success:
                        continue
                        
                    if len(processed_photos) == 1:
                        f_name = f"{student_id}_{student_name}.jpg"
                    else:
                        f_name = f"{student_id}_{student_name}({idx + 1}).jpg"
                        
                    upload_via_gas(buffer.tobytes(), f_name, target_gas_url)
                    uploaded_names.append((f_name, photo))
                
                if uploaded_names:
                    st.success(f"🎉 [{student_id} {student_name}] 학생의 증명사진 {len(uploaded_names)}장이 성공적으로 추출 및 저장되었습니다!")
                    
                    cols = st.columns(len(uploaded_names))
                    for col, (f_name, p_img) in zip(cols, uploaded_names):
                        with col:
                            st.image(p_img, channels="BGR", caption=f"저장 완료: {f_name}")
                else:
                    st.error("사진 변환 및 업로드에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")

# 학생 화면 하단 관리자 복귀 링크
if current_code and not st.session_state.get("is_teacher_creator", False):
    st.markdown("<br><hr><center><small style='color: gray;'>선생님이신가요? <a href='.' style='color: gray; text-decoration: underline;'>새 링크 만들기</a></small></center>", unsafe_allow_html=True)
