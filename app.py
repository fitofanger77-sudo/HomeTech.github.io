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
    """4개의 꼭짓점을 시계 방향(좌상, 우상, 우하, 좌하)으로 정렬하고 증명사진 세로 방향 유지"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    ordered = np.array([tl, tr, br, bl], dtype="float32")
    
    # 가로/세로 판별 후 세로 방향으로 회전 보정
    w = (np.linalg.norm(ordered[0] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[2])) / 2.0
    h = (np.linalg.norm(ordered[1] - ordered[2]) + np.linalg.norm(ordered[0] - ordered[3])) / 2.0
    if w > h:
        ordered = np.roll(ordered, -1, axis=0)
        
    return ordered

def get_true_quad(contour):
    """인화지의 실제 4개 꼭짓점(빨간 점)을 정밀 검출"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    
    for eps in [0.03, 0.04, 0.02, 0.05, 0.015]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
            
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def is_actual_portrait_photo(cropped_img):
    """자물쇠, 플라스틱 부품 등 무생물을 걸러내고 실제 사람 얼굴 사진인지 검증"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # 얼굴이 집중된 중앙 상단 영역 (세로 20%~70%, 가로 25%~75%)
    roi = hsv[int(h * 0.20):int(h * 0.70), int(w * 0.25):int(w * 0.75)]
    
    lower_skin1 = np.array([0, 28, 55], dtype=np.uint8)
    upper_skin1 = np.array([25, 185, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 28, 55], dtype=np.uint8)
    upper_skin2 = np.array([180, 185, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(roi, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(roi, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    skin_ratio = np.sum(skin_mask > 0) / (roi.shape[0] * roi.shape[1])
    
    # 채도 분석: 자물쇠/회색 플라스틱(채도 낮음) 대비 자연스러운 피부/옷(채도 높음)
    sat_mean = hsv[:, :, 1].mean()
    
    return (skin_ratio >= 0.15 and sat_mean >= 25.0)

def extract_all_photos(image):
    """실제 인물 사진만 추출하고 각 사진의 4개 꼭짓점을 기준으로 수평 정렬"""
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    # 1. 순수 디지털 파일 검사 (가장자리 평균 밝기)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    if np.mean(border_concat) > 140:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh_bright = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    c1, _ = cv2.findContours(closed_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(closed_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts = list(c1) + list(c2)
    
    candidates = []
    for c in all_cnts:
        area = cv2.contourArea(c)
        if not (img_area * 0.008 < area < img_area * 0.40):
            continue
            
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        if not (0.60 < ar < 0.95):
            continue
            
        quad_pts = get_true_quad(c)
        
        peri = cv2.arcLength(cv2.convexHull(c), True)
        approx = cv2.approxPolyDP(cv2.convexHull(c), 0.03 * peri, True)
        is_quad = 1.0 if len(approx) == 4 else 0.0
        
        score = is_quad * 50.0 + (area / img_area) * 100.0
        candidates.append({
            'pts': quad_pts,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area,
            'score': score
        })
        
    if not candidates:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    # NMS (중복 영역 제거)
    candidates.sort(key=lambda x: x['score'], reverse=True)
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
            distinct.append(cand)
            
    # 정렬: 위에서 아래, 좌에서 우
    distinct.sort(key=lambda x: (x['center'][1] // 200, x['center'][0]))
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    verified_photos = []
    for p in distinct:
        ordered_pts = robust_order_points(p['pts'])
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
        
        # 실제 인물 사진 검증
        if is_actual_portrait_photo(warped):
            trimmed = warped[2:target_h-2, 2:target_w-2]
            final_p = cv2.resize(trimmed, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            verified_photos.append(final_p)
            
    if not verified_photos and distinct:
        ordered_pts = robust_order_points(distinct[0]['pts'])
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
        verified_photos.append(warped)
        
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
