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
    """카메라 촬영 기준에 맞추어 4개 꼭짓점을 [좌상, 우상, 우하, 좌하]로 정확히 매핑"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    return np.array([tl, tr, br, bl], dtype="float32")

def get_true_quad(contour):
    """인화지의 실제 4개 모서리 꼭짓점 검출"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in [0.03, 0.04, 0.02, 0.05, 0.015]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def refine_and_deskew(warped, target_w=350, target_h=450):
    """미세한 기울기(각도 오차)만 수평 0도로 정밀 보정 (상하반전 금지)"""
    if warped.shape[0] != target_h or warped.shape[1] != target_w:
        warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_w, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=60, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for l in lines:
            line_arr = np.array(l).flatten()
            if len(line_arr) >= 4:
                x1, y1, x2, y2 = line_arr[:4]
                deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(deg) < 12:
                    angles.append(deg)
                elif abs(deg - 180) < 12:
                    angles.append(deg - 180)
                elif abs(deg + 180) < 12:
                    angles.append(deg + 180)
                elif abs(abs(deg) - 90) < 12:
                    angles.append(deg - 90 if deg > 0 else deg + 90)
                
    # 5도 이내의 미세 회전 오차만 정밀 평행 보정
    if len(angles) >= 2:
        tilt = float(np.median(angles))
        if 0.25 <= abs(tilt) < 5.0:
            rot_M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt, 1.0)
            warped = cv2.warpAffine(warped, rot_M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    # 외곽 바닥면 잔상 정리 (2px 안전 마진)
    trim = 2
    final = warped[trim:target_h-trim, trim:target_w-trim]
    return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def is_actual_portrait_photo(cropped_img):
    """엄격한 증명사진 최종 자가 검증 (피부톤 + 이목구비 + 명암)"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # [차단 1] 시커먼 암전 이미지, 극단적 단색 배제
    overall_mean = gray.mean()
    overall_std = gray.std()
    if overall_mean < 65 or overall_mean > 240 or overall_std < 22:
        return False
        
    # [차단 2] 중심부 사람 피부톤 검증
    roi_hsv = hsv[int(h * 0.15):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    lower_skin1 = np.array([0, 25, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 190, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 25, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 190, 255], dtype=np.uint8)
    mask1 = cv2.inRange(roi_hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(roi_hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_ratio = np.sum(skin_mask > 0) / (roi_hsv.shape[0] * roi_hsv.shape[1])
    if skin_ratio < 0.08:
        return False
        
    # [차단 3] 눈·코·입 등 이목구비 윤곽선 검증 (평평한 나무책상/플라스틱 배제)
    gray_roi = gray[int(h * 0.15):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_count = (edges > 0).sum()
    if edge_count < 100:
        return False
        
    return True

def extract_all_photos(image):
    """다단계 적응형 탐색 및 정밀 보정 엔진"""
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    # 1. 순수 디지털 원본 파일인지 검사
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    if np.mean(border_concat) > 140:
        res = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
        return [res] if is_actual_portrait_photo(res) else []
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 조명/그림자에 구애받지 않는 다중 채널 윤곽선 추출
    _, thresh_bright = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    edged = cv2.Canny(blur, 35, 120)
    
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_e)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    
    c1, _ = cv2.findContours(closed_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(closed_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c3, _ = cv2.findContours(closed_adapt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c4, _ = cv2.findContours(edged_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts = list(c1) + list(c2) + list(c3) + list(c4)
    
    candidates = []
    for c in all_cnts:
        area = cv2.contourArea(c)
        if not (img_area * 0.005 < area < img_area * 0.45):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        if not (0.58 < ar < 0.98):
            continue
            
        quad_pts = get_true_quad(c)
        peri = cv2.arcLength(cv2.convexHull(c), True)
        approx = cv2.approxPolyDP(cv2.convexHull(c), 0.03 * peri, True)
        is_quad = 1.0 if len(approx) == 4 else 0.0
        
        ratio_diff = min(abs(ar - 0.75), abs(ar - 0.778))
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.15)) * 50.0
        score = is_quad * 50.0 + ratio_score + (area / img_area) * 80.0
        
        candidates.append({
            'pts': quad_pts,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area,
            'score': score
        })
        
    candidates.sort(key=lambda x: x['score'], reverse=True)
    distinct = []
    for cand in candidates:
        cx1, cy1 = cand['center']
        is_overlap = False
        for kept in distinct:
            dist = np.hypot(cx1 - kept['center'][0], cy1 - kept['center'][1])
            if dist < min(cand['w'], kept['w']) * 0.65:
                is_overlap = True
                break
        if not is_overlap:
            distinct.append(cand)
            
    # 좌상단부터 순서대로 정렬
    distinct.sort(key=lambda x: (x['center'][1] // 200, x['center'][0]))
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    verified_photos = []
    for p in distinct:
        ordered_pts = robust_order_points(p['pts'])
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
        refined = refine_and_deskew(warped, target_w, target_h)
        
        # [철저한 검증] 증명사진으로 완벽히 판정된 사진만 최종 통과
        if is_actual_portrait_photo(refined):
            verified_photos.append(refined)
            
    # 통과한 사진이 없으면 절대로 잔상/잡동사니를 내보내지 않고 빈 리스트 반환
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
        with st.spinner("사진을 분석하고 정밀 스캔 중입니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # 정밀 추출 및 검증 엔진 실행
                processed_photos = extract_all_photos(image)
                
                # [게이트키퍼] 진짜 증명사진 판정이 안 나면 절대 드라이브 전송하지 않고 에러 출력!
                if not processed_photos:
                    st.error("🚨 **사진 속에서 규격에 맞는 증명사진을 감지하지 못했습니다!**\n\n- 빛 반사가 심하거나 증명사진이 너무 작게 찍혔습니다.\n- 사진이 잘 보이도록 조금 더 가까이 촬영한 후 다시 업로드해 주세요.")
                else:
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
