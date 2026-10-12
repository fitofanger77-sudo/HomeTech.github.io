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
    
    # 세로 방향 비율 유지
    w = (np.linalg.norm(ordered[0] - ordered[1]) + np.linalg.norm(ordered[3] - ordered[2])) / 2.0
    h = (np.linalg.norm(ordered[1] - ordered[2]) + np.linalg.norm(ordered[0] - ordered[3])) / 2.0
    if w > h:
        ordered = np.roll(ordered, -1, axis=0)
        
    return ordered

def get_true_quad(contour):
    """인화지의 실제 4개 꼭짓점(모서리)을 정밀 검출"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    
    for eps in [0.03, 0.04, 0.02, 0.05, 0.015]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
            
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype("float32")

def refine_and_deskew(warped, target_w=350, target_h=450):
    """결과물을 재점검하고 삐뚤어 보이면 각도와 테두리를 2차 정밀 보정 (안전 언패킹)"""
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
                if abs(deg) < 15:
                    angles.append(deg)
                elif abs(deg - 180) < 15:
                    angles.append(deg - 180)
                elif abs(deg + 180) < 15:
                    angles.append(deg + 180)
                elif abs(abs(deg) - 90) < 15:
                    angles.append(deg - 90 if deg > 0 else deg + 90)
                
    if len(angles) >= 2:
        tilt = float(np.median(angles))
        if 0.25 <= abs(tilt) < 10.0:
            rot_M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt, 1.0)
            warped = cv2.warpAffine(warped, rot_M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    corners = [
        warped[0:10, 0:10],
        warped[0:10, target_w-10:target_w],
        warped[target_h-10:target_h, 0:10],
        warped[target_h-10:target_h, target_w-10:target_w]
    ]
    corner_means = [c.mean() for c in corners]
    trim = 5 if min(corner_means) < 110 else 2
    
    final = warped[trim:target_h-trim, trim:target_w-trim]
    return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def is_actual_portrait_photo(cropped_img):
    """사물/책상/자물쇠를 배제하고 실제 사람 얼굴(피부톤 + 이목구비 윤곽) 사진인지 엄격 검증"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[int(h * 0.15):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    
    # 피부톤 영역
    lower_skin1 = np.array([0, 25, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 190, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 25, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 190, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(roi_hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(roi_hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_ratio = np.sum(skin_mask > 0) / (roi_hsv.shape[0] * roi_hsv.shape[1])
    
    # 이목구비 윤곽선(Canny) 밀도 검증 (평평한 나무판자 차단)
    gray_roi = cv2.cvtColor(cropped_img[int(h * 0.15):int(h * 0.75), int(w * 0.20):int(w * 0.80)], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_count = (edges > 0).sum()
    
    # 피부색 비율 10% 이상, 눈코입 윤곽선 150픽셀 이상 동시 충족
    return (skin_ratio >= 0.10 and edge_count >= 150)

def search_candidates_from_contours(all_cnts, img_h, img_w):
    """윤곽선 리스트에서 증명사진 후보 박스 추출"""
    img_area = img_h * img_w
    candidates = []
    for c in all_cnts:
        area = cv2.contourArea(c)
        if not (img_area * 0.005 < area < img_area * 0.50):
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
        
        ratio_diff = min(abs(ar - 0.75), abs(ar - 0.778))
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.15)) * 50.0
        score = is_quad * 50.0 + ratio_score + (area / img_area) * 100.0
        
        candidates.append({
            'pts': quad_pts,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area,
            'score': score
        })
    return candidates

def extract_all_photos(image):
    """다단계 재검토 및 형광등 반사 자동 극복 엔진"""
    img_h, img_w = image.shape[:2]
    
    # 1. 순수 디지털 원본 파일인지 검사 (가장자리 평균 밝기)
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    if np.mean(border_concat) > 140:
        return [cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)]
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    # ==========================================
    # [1단계 탐색] 밝기 및 이진화 기반 스캔
    # ==========================================
    _, thresh_bright = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    c1, _ = cv2.findContours(closed_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(closed_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = search_candidates_from_contours(list(c1) + list(c2), img_h, img_w)
    
    # 1단계 후보군 중 실제 인물 사진인지 자가 점검
    verified_photos = []
    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        distinct = []
        for cand in candidates:
            cx1, cy1 = cand['center']
            is_overlap = False
            for kept in distinct:
                dist = np.hypot(cx1 - kept['center'][0], cy1 - kept['center'][1])
                if dist < min(cand['w'], kept['w']) * 0.7:
                    is_overlap = True
                    break
            if not is_overlap:
                distinct.append(cand)
                
        for p in distinct:
            ordered_pts = robust_order_points(p['pts'])
            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
            refined = refine_and_deskew(warped, target_w, target_h)
            if is_actual_portrait_photo(refined):
                verified_photos.append(refined)

    # ==========================================
    # [2단계 재탐색] 1단계 실패 시: 형광등 반사를 분리하는 엣지(Edge) 전면 재스캔
    # ==========================================
    if not verified_photos:
        edged = cv2.Canny(blur, 40, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edge = cv2.dilate(edged, kernel_edge, iterations=1)
        cnts_edge, _ = cv2.findContours(dilated_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates_edge = search_candidates_from_contours(cnts_edge, img_h, img_w)
        if candidates_edge:
            candidates_edge.sort(key=lambda x: x['score'], reverse=True)
            for p in candidates_edge:
                ordered_pts = robust_order_points(p['pts'])
                M = cv2.getPerspectiveTransform(ordered_pts, dst)
                warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
                refined = refine_and_deskew(warped, target_w, target_h)
                if is_actual_portrait_photo(refined):
                    verified_photos.append(refined)
                    break

    # ==========================================
    # [3단계 재탐색] 2단계 실패 시: 피부톤 클러스터 기준 역추적
    # ==========================================
    if not verified_photos:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin1 = np.array([0, 25, 50], dtype=np.uint8)
        upper_skin1 = np.array([25, 190, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin1, upper_skin1)
        kernel_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_skin, iterations=3)
        cnts_skin, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cs in cnts_skin:
            if cv2.contourArea(cs) > 2000:
                rect_s = cv2.minAreaRect(cs)
                scx, scy = rect_s[0]
                sw, sh = rect_s[1]
                # 얼굴 크기 기준 추정 마진 (3.5x4.5 비율)
                box_h = int(max(sw, sh) * 2.3)
                box_w = int(box_h * 0.77)
                y1 = max(0, int(scy - box_h * 0.45))
                y2 = min(img_h, int(y1 + box_h))
                x1 = max(0, int(scx - box_w * 0.5))
                x2 = min(img_w, int(x1 + box_w))
                
                face_sub = image[y1:y2, x1:x2]
                if face_sub.shape[0] > 50 and face_sub.shape[1] > 50:
                    cand_img = cv2.resize(face_sub, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    cand_refined = refine_and_deskew(cand_img, target_w, target_h)
                    if is_actual_portrait_photo(cand_refined):
                        verified_photos.append(cand_refined)
                        break
                        
    # 최종적으로도 증명사진이 아니면 빈 리스트 반환 (책상 전체 전송 차단)
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
        with st.spinner("사진을 정밀 분석하고 자동 보정 스캔 중입니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # 다단계 재검토 추출 엔진 실행
                processed_photos = extract_all_photos(image)
                
                # [게이트키퍼] 진짜 증명사진으로 검증되지 않은 경우 전송 차단
                if not processed_photos:
                    st.error("🚨 **증명사진 영역을 감지하지 못했습니다!**\n\n형광등 빛 반사가 심하거나 사진이 너무 멀리 떨어져 있습니다. **빛 반사가 없는 곳에서 사진을 조금 더 가까이 촬영하여 다시 업로드해 주세요.**")
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
