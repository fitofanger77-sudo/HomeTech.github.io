import streamlit as st
import cv2
import numpy as np
import os
import io
import base64
import requests

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

# 세션 상태 초기화
if "scan_candidates" not in st.session_state:
    st.session_state["scan_candidates"] = None
if "selected_idx" not in st.session_state:
    st.session_state["selected_idx"] = 0
if "rotation_deg" not in st.session_state:
    st.session_state["rotation_deg"] = 0
if "student_meta" not in st.session_state:
    st.session_state["student_meta"] = ("", "")
if "final_uploaded" not in st.session_state:
    st.session_state["final_uploaded"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def reset_scan_state():
    """재촬영/재업로드를 위해 스캔 상태를 초기화하는 함수"""
    st.session_state["scan_candidates"] = None
    st.session_state["selected_idx"] = 0
    st.session_state["rotation_deg"] = 0
    st.session_state["final_uploaded"] = None
    st.session_state["uploader_key"] += 1

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
st.write("학번과 이름을 입력하고 증명사진을 등록하면, 스캔 결과를 먼저 확인한 뒤 선생님께 안전하게 제출할 수 있습니다.")

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

# --- 1단계: 학생 정보 및 업로드 방식 선택 ---
st.subheader("1️⃣ 학번·이름 입력 및 사진 등록 방식 선택")
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("학번 (예: 1학년 1반 1번인 경우, 10101)")
with col2:
    student_name = st.text_input("이름 (예: 홍길동)")

upload_mode = st.radio(
    "증명사진 등록 방식을 선택해 주세요:",
    options=[
        "📸 실물 증명사진 촬영본 올리기 (AI 자동 스캔·테두리 맞춤)",
        "📷 카메라로 지금 바로 촬영하기 (AI 자동 스캔·테두리 맞춤)",
        "🖼️ 디지털 증명사진 원본 파일 올리기 (스캔 없이 규격만 맞춤)"
    ],
    index=0
)

input_file_buffer = None
is_direct_digital_mode = ("디지털 증명사진 원본" in upload_mode)

if "카메라로 지금 바로 촬영" in upload_mode:
    st.caption("💡 빛 반사나 그림자가 없는 어두운 바닥 위에 증명사진을 놓고 정면에서 촬영해 주세요.")
    input_file_buffer = st.camera_input(
        "증명사진 촬영",
        key=f"cam_{st.session_state['uploader_key']}"
    )
elif is_direct_digital_mode:
    st.caption("💡 사진관에서 받은 이미지 파일이나 이미 여백 없이 잘라진 증명사진 파일을 선택해 주세요.")
    input_file_buffer = st.file_uploader(
        "디지털 증명사진 원본 파일 선택",
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
        key=f"dig_{st.session_state['uploader_key']}"
    )
else:
    st.caption("💡 책상이나 바닥 위에 실물 종이 증명사진을 놓고 스마트폰으로 찍은 사진을 선택해 주세요.")
    input_file_buffer = st.file_uploader(
        "실물 증명사진 촬영본 파일 선택",
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
        key=f"phy_{st.session_state['uploader_key']}"
    )

# ==========================================================
# 영상 처리 및 정밀 검증 알고리즘
# ==========================================================
def robust_order_points(pts):
    """45도 이상 기울어진 사진도 짧은 변(위/아래)과 긴 변(좌/우)을 구분하여 [좌상, 우상, 우하, 좌하]로 정렬"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    poly = pts[np.argsort(angles)]
    
    l0 = np.linalg.norm(poly[0] - poly[1])
    l1 = np.linalg.norm(poly[1] - poly[2])
    l2 = np.linalg.norm(poly[2] - poly[3])
    l3 = np.linalg.norm(poly[3] - poly[0])
    
    mean_02 = (l0 + l2) / 2.0
    mean_13 = (l1 + l3) / 2.0
    
    if mean_02 <= mean_13:
        cand_a = poly
        cand_b = np.roll(poly, -2, axis=0)
    else:
        cand_a = np.roll(poly, -1, axis=0)
        cand_b = np.roll(poly, -3, axis=0)
        
    top_y_a = (cand_a[0][1] + cand_a[1][1]) / 2.0
    top_y_b = (cand_b[0][1] + cand_b[1][1]) / 2.0
    
    if top_y_a <= top_y_b:
        return cand_a.astype("float32")
    else:
        return cand_b.astype("float32")

def get_candidate_quads(contour):
    """윤곽선으로부터 4개 꼭짓점 후보 및 직각 모서리 복원 후보 추출"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    quads = []
    
    for eps in [0.02, 0.025, 0.03, 0.04]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ordered = robust_order_points(approx.reshape(4, 2).astype("float32"))
            quads.append(ordered)
            
            # 빛 반사로 1개 모서리가 바깥으로 튄 경우 반대쪽 3개 모서리로 평행사변형 복원
            orig_area = cv2.contourArea(ordered)
            for k in range(4):
                opp = (k + 2) % 4
                fixed = ordered.copy()
                fixed[k] = ordered[(k - 1) % 4] + ordered[(k + 1) % 4] - ordered[opp]
                if cv2.isContourConvex(fixed):
                    f_area = cv2.contourArea(fixed)
                    if orig_area * 0.72 <= f_area <= orig_area * 1.05:
                        quads.append(robust_order_points(fixed))
            break
            
    rect = cv2.minAreaRect(hull)
    quads.append(robust_order_points(cv2.boxPoints(rect).astype("float32")))
    return quads

def gentle_deskew(warped, target_w=350, target_h=450):
    """이미지를 훼손하지 않고 미세한 수평 기울기(4도 이내)와 외곽 4픽셀 테두리만 깔끔히 정리"""
    if warped.shape[0] != target_h or warped.shape[1] != target_w:
        warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_w, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=70, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for l in lines:
            line_arr = np.array(l).flatten()
            if len(line_arr) >= 4:
                x1, y1, x2, y2 = line_arr[:4]
                deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(deg) < 8:
                    angles.append(deg)
                elif abs(deg - 180) < 8:
                    angles.append(deg - 180)
                elif abs(deg + 180) < 8:
                    angles.append(deg + 180)
                elif abs(abs(deg) - 90) < 8:
                    angles.append(deg - 90 if deg > 0 else deg + 90)
                    
    if len(angles) >= 2:
        tilt = float(np.median(angles))
        if 0.25 <= abs(tilt) < 4.5:
            rot_M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt, 1.0)
            warped = cv2.warpAffine(warped, rot_M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    trim_x = 5
    trim_y = 4
    final = warped[trim_y:target_h-trim_y, trim_x:target_w-trim_x]
    return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def get_skin_mask(hsv_img):
    """HSV 이미지에서 사람 피부톤 영역 마스크 반환"""
    lower_skin1 = np.array([0, 22, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 190, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 22, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 190, 255], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lower_skin1, upper_skin1),
        cv2.inRange(hsv_img, lower_skin2, upper_skin2)
    )

def validate_and_score_id_photo(cropped_img):
    """
    [엄격한 증명사진 판별기]
    코/입만 확대된 사진, 책상이 끼어든 사다리꼴 사진, 자물쇠/배경 등을 철저히 탈락시킴.
    반환값: (통과여부_bool, 품질점수_float)
    """
    h, w = cropped_img.shape[:2]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY).astype("float32")
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    skin_mask = get_skin_mask(hsv)
    
    # 1. 전체 밝기 및 대비 검사 (시커먼 배경이나 단색 판자 탈락)
    overall_mean = float(gray.mean())
    overall_std = float(gray.std())
    if overall_mean < 65.0 or overall_mean > 240.0 or overall_std < 24.0:
        return False, -1e9
        
    # 2. 중심부 피부톤 비율 검사 (얼굴이 아예 없거나, 반대로 코/입만 확대되어 화면을 덮은 경우 탈락!)
    center_skin = skin_mask[int(h * 0.20):int(h * 0.72), int(w * 0.22):int(w * 0.78)]
    center_skin_ratio = float((center_skin > 0).mean())
    overall_skin_ratio = float((skin_mask > 0).mean())
    
    # 코/입 클로즈업 사진은 전체 피부 비율이 50%를 넘거나 중앙 피부 비율이 70%를 넘음 -> 즉시 탈락!
    if center_skin_ratio < 0.08 or center_skin_ratio > 0.68 or overall_skin_ratio > 0.48:
        return False, -1e9
        
    # 3. 테두리(상단/하단/좌우 끝) 피부 침범 검사
    # 정상적인 증명사진은 사진 맨 위쪽(정수리 위 여백)이나 좌/우 끝 테두리에 살색(피부)이 닿지 않음!
    top_edge_skin = float((skin_mask[0:int(h * 0.10), :] > 0).mean())
    left_edge_skin = float((skin_mask[:, 0:int(w * 0.08)] > 0).mean())
    right_edge_skin = float((skin_mask[:, int(w * 0.92):] > 0).mean())
    if top_edge_skin > 0.12 or left_edge_skin > 0.20 or right_edge_skin > 0.20:
        return False, -1e9
        
    # 4. 상단 좌/우 배경 모서리 균일성 및 대칭성 검사 (책상 바닥이 한쪽 모서리에 끼어든 사진 탈락!)
    tl_gray = gray[8:55, 8:55]
    tr_gray = gray[8:55, w-55:w-8]
    tl_sat = hsv[8:55, 8:55, 1].astype("float32")
    tr_sat = hsv[8:55, w-55:w-8, 1].astype("float32")
    
    # 상단 모서리가 어둡거나(갈색 책상 등), 좌우 모서리 밝기 차이가 심하면(사다리꼴 왜곡) 탈락
    if float(tl_gray.mean()) < 95.0 or float(tr_gray.mean()) < 95.0:
        return False, -1e9
    lr_bright_diff = abs(float(tl_gray.mean()) - float(tr_gray.mean()))
    if lr_bright_diff > 48.0:
        return False, -1e9
        
    # 5. 정수리 위 여백(Headroom) 확인 (머리카락이 천장을 완전히 덮어 얼굴만 잘린 사진 탈락)
    top_center_patch = gray[4:28, int(w * 0.25):int(w * 0.75)]
    if float(np.percentile(top_center_patch, 80)) < 110.0:
        return False, -1e9
        
    # 6. 중심부 이목구비 윤곽선 밀도 확인
    gray_roi = cv2.cvtColor(cropped_img[int(h * 0.18):int(h * 0.75), int(w * 0.20):int(w * 0.80)], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 45, 140)
    if (edges > 0).sum() < 110:
        return False, -1e9
        
    # 품질 점수 계산: 상단 좌/우 모서리가 밝고 대칭적이며 채도가 낮을수록 높은 점수 부여
    corner_bright = (float(tl_gray.mean()) + float(tr_gray.mean())) / 2.0
    corner_sat = (float(tl_sat.mean()) + float(tr_sat.mean())) / 2.0
    corner_std = (float(tl_gray.std()) + float(tr_gray.std())) / 2.0
    
    quality_score = (
        corner_bright * 1.2
        - lr_bright_diff * 2.0
        - corner_sat * 1.2
        - corner_std * 1.5
    )
    return True, quality_score

def scan_physical_photos(image):
    """실물 촬영 사진에서 온전한 증명사진만 정밀 스캔 (불량 크롭은 자동 폐기)"""
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    # 만약 학생이 실물 모드에 순수 디지털 증명사진 파일을 올린 경우 자동 호환
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    if np.mean(border_concat) > 155:
        resized = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
        ok, _ = validate_and_score_id_photo(resized)
        if ok:
            return [resized]
            
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1) 로컬 경계선(적응형 이진화 + Canny) 결합 마스크 -> 형광등 빛 반사를 무시하고 종이 테두리만 검출!
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 140)
    combined_edge = cv2.bitwise_or(thresh_adapt, edged)
    
    # 2) 밝기 및 저채도 인화지 마스크
    paper_hsv = cv2.inRange(hsv_full, np.array([0, 0, 140], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8))
    _, thresh_bright = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edge = cv2.morphologyEx(combined_edge, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_paper = cv2.morphologyEx(paper_hsv, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    
    all_cnts = []
    for m_img in [closed_paper, closed_bright, closed_otsu]:
        cnts, _ = cv2.findContours(m_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)
    cnts_e, _ = cv2.findContours(closed_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts.extend(cnts_e)
    
    raw_items = []
    for c in all_cnts:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        # 너무 작은 내부 조각(코/입 등, 전체 화면의 1.5% 미만)은 아예 후보에서 원천 차단!
        if not (img_area * 0.015 < area < img_area * 0.65):
            continue
        rect = cv2.minAreaRect(hull)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        if not (0.62 < ar < 0.94):
            continue
            
        raw_items.append({
            'cnt': hull,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area
        })
        
    if not raw_items:
        return []
        
    # 큰 외곽 인화지부터 정렬
    raw_items.sort(key=lambda x: x['area'], reverse=True)
    
    # 위치별 클러스터링 (동일 사진의 중복 윤곽선 묶기)
    clusters = []
    for item in raw_items:
        cx1, cy1 = item['center']
        placed = False
        for cl in clusters:
            cx2, cy2 = cl[0]['center']
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < max(item['w'], cl[0]['w']) * 0.65:
                cl.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
            
    clusters.sort(key=lambda cl: (cl[0]['center'][1] // 200, cl[0]['center'][0]))
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    verified_photos = []
    for cl in clusters:
        max_a = max(x['area'] for x in cl)
        # 클러스터 내에서 가장 큰 외곽선 대비 70% 이상 크기를 가진 외곽 후보만 검사 (내부 이목구비 크롭 원천 차단)
        outer_candidates = [x for x in cl if x['area'] >= max_a * 0.70]
        
        best_img = None
        best_score = -1e9
        
        for item in outer_candidates[:5]:
            quads = get_candidate_quads(item['cnt'])
            for q_pts in quads:
                ordered_pts = robust_order_points(q_pts)
                M = cv2.getPerspectiveTransform(ordered_pts, dst)
                warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
                refined = gentle_deskew(warped, target_w, target_h)
                
                is_valid, q_score = validate_and_score_id_photo(refined)
                if is_valid and q_score > best_score:
                    best_score = q_score
                    best_img = refined
                    
        if best_img is not None:
            verified_photos.append(best_img)
            
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

# --- 스캔 및 미리보기 실행 버튼 ---
if st.button("🔍 사진 스캔 및 결과 확인하기", type="primary", use_container_width=True):
    st.session_state["scan_candidates"] = None
    st.session_state["selected_idx"] = 0
    st.session_state["rotation_deg"] = 0
    st.session_state["final_uploaded"] = None
    
    if not target_gas_url:
        st.error("🚨 담임 선생님의 구글 드라이브 연결 정보가 없습니다! 담임 선생님께서 공유해주신 전용 링크로 다시 접속해주세요.")
    elif not student_id.strip() or not student_name.strip():
        st.warning("⚠️ 학번과 이름을 모두 입력해주세요!")
    elif input_file_buffer is None:
        st.warning("⚠️ 먼저 증명사진을 촬영하거나 파일을 선택해주세요!")
    else:
        with st.spinner("사진을 정밀 분석하고 증명사진 규격을 확인 중입니다..."):
            try:
                file_bytes = np.frombuffer(input_file_buffer.getvalue(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if is_direct_digital_mode:
                    # 디지털 원본 파일 모드: 왜곡 없이 350x450 규격 리사이즈
                    resized_digital = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
                    st.session_state["scan_candidates"] = [resized_digital]
                    st.session_state["student_meta"] = (student_id.strip(), student_name.strip())
                else:
                    # 실물 증명사진 스캔 모드: 엄격한 증명사진 구조 검증 통과본만 반환
                    scanned_list = scan_physical_photos(image)
                    if not scanned_list:
                        st.error(
                            "🚨 **증명사진의 전체 테두리를 정확하게 인식하지 못했습니다! 다시 촬영해 주세요.**\n\n"
                            "**[재촬영 팁]**\n"
                            "1. 천장 형광등 불빛이 사진에 직접 반사되지 않도록 각도를 살짝 조절해 주세요.\n"
                            "2. 어두운 책상이나 마우스패드 위에 사진을 놓고, **증명사진의 네 모서리가 모두 화면에 또렷하게 나오도록** 다시 찍어 주세요.\n"
                            "3. 만약 사진관에서 받은 원본 파일이라면 위에서 **'🖼️ 디지털 증명사진 원본 파일 올리기'**를 선택해 주세요."
                        )
                    else:
                        st.session_state["scan_candidates"] = scanned_list
                        st.session_state["student_meta"] = (student_id.strip(), student_name.strip())
            except Exception as e:
                st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")

# --- 2단계: 스캔 결과 확인 및 학생 최종 제출 결정 단계 ---
if st.session_state["scan_candidates"] is not None:
    st.markdown("---")
    st.subheader("2️⃣ 스캔 결과 확인 및 최종 제출 여부 선택")
    
    cands = st.session_state["scan_candidates"]
    s_id, s_name = st.session_state["student_meta"]
    
    # 여러 장의 사진이 감지된 경우 학생이 먼저 자기 사진을 선택
    if len(cands) > 1:
        st.info(f"🧐 사진 속에서 총 **{len(cands)}장**의 사진이 감지되었습니다. 먼저 제출할 본인의 사진을 선택해 주세요.")
        cols = st.columns(len(cands))
        for idx, (col, p_img) in enumerate(zip(cols, cands)):
            with col:
                is_sel = (st.session_state["selected_idx"] == idx)
                st.image(p_img, channels="BGR", caption=f"후보 사진 {idx + 1}")
                btn_label = f"✔️ {idx + 1}번 선택됨" if is_sel else f"{idx + 1}번 사진 선택"
                if st.button(btn_label, key=f"pick_{idx}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    st.session_state["selected_idx"] = idx
                    st.session_state["rotation_deg"] = 0
                    st.rerun()

    # 현재 선택된 사진 가져오기 및 수동 회전 적용
    current_img = cands[st.session_state["selected_idx"]].copy()
    rot_k = (st.session_state["rotation_deg"] // 90) % 4
    if rot_k != 0:
        current_img = np.rot90(current_img, -rot_k) # 시계 방향 회전
        current_img = cv2.resize(current_img, (350, 450), interpolation=cv2.INTER_CUBIC)

    st.warning("📢 **잠깐! 아직 제출되지 않았습니다.** 아래 스캔 결과물이 잘리거나 삐뚤어지지 않았는지 확인한 후 최종 제출을 눌러주세요.")
    
    prev_col1, prev_col2 = st.columns([1, 1.3])
    with prev_col1:
        st.image(current_img, channels="BGR", caption=f"제출 예정 파일: {s_id}_{s_name}.jpg", width=260)
        if st.button("🔃 사진이 누워있나요? (시계방향 90° 회전)", use_container_width=True):
            st.session_state["rotation_deg"] = (st.session_state["rotation_deg"] + 90) % 360
            st.rerun()
            
    with prev_col2:
        st.markdown(
            f"**📋 제출 정보 확인**\n"
            f"- **저장될 파일명:** `{s_id}_{s_name}.jpg`\n"
            f"- **사진 규격:** 표준 증명사진 (350 × 450 px)\n\n"
            f"**확인 사항:**\n"
            f"1. 얼굴이나 어깨가 잘리지 않고 증명사진 전체가 잘 나왔나요?\n"
            f"2. 사진이 심하게 찌그러지거나 빛 반사로 가려진 곳은 없나요?\n\n"
            f"결과물이 마음에 들지 않으면 **재촬영 / 다시 업로드** 버튼을 눌러주세요."
        )
        
        st.write("")
        if st.button("✅ 네, 이 스캔 결과물로 최종 제출합니다!", type="primary", use_container_width=True):
            with st.spinner("선생님의 구글 드라이브로 증명사진을 안전하게 전송 중입니다..."):
                try:
                    is_success, buffer = cv2.imencode(".jpg", current_img)
                    if is_success:
                        f_name = f"{s_id}_{s_name}.jpg"
                        upload_via_gas(buffer.tobytes(), f_name, target_gas_url)
                        st.session_state["final_uploaded"] = (f_name, current_img)
                        st.session_state["scan_candidates"] = None
                        st.rerun()
                    else:
                        st.error("이미지 인코딩 중 오류가 발생했습니다.")
                except Exception as e:
                    st.error(f"구글 드라이브 전송 중 오류가 발생했습니다: {e}")
                    
        if st.button("🔄 아니오, 사진을 다시 촬영 / 재업로드할게요", type="secondary", use_container_width=True):
            reset_scan_state()
            st.rerun()

# --- 3단계: 최종 업로드 완료 화면 ---
if st.session_state["final_uploaded"] is not None:
    st.markdown("---")
    f_name, final_img = st.session_state["final_uploaded"]
    st.success(f"🎉 **[{f_name}]** 증명사진이 선생님 구글 드라이브에 최종 제출 완료되었습니다! 수고하셨습니다.")
    st.image(final_img, channels="BGR", caption=f"최종 제출 완료된 사진 ({f_name})", width=250)
    if st.button("다른 사진 다시 제출하기"):
        reset_scan_state()
        st.rerun()

# 학생 화면 하단 관리자 복귀 링크
if current_code and not st.session_state.get("is_teacher_creator", False):
    st.markdown("<br><hr><center><small style='color: gray;'>선생님이신가요? <a href='.' style='color: gray; text-decoration: underline;'>새 링크 만들기</a></small></center>", unsafe_allow_html=True)
