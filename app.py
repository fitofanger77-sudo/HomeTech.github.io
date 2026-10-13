import streamlit as st
import cv2
import numpy as np
import os
import io
import base64
import requests

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

# 세션 상태 초기화 (다중 사진 선택 및 완료 상태 유지용)
if "candidates" not in st.session_state:
    st.session_state["candidates"] = None
if "student_meta" not in st.session_state:
    st.session_state["student_meta"] = ("", "")
if "upload_result" not in st.session_state:
    st.session_state["upload_result"] = None

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

def get_true_quads(contour):
    """인화지 4개 모서리 추출 및 빛 반사로 삐져나간 단일 모서리의 평행사변형 기하학 복원 후보 생성"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    quads = []
    
    # 1. 최소외접사각형 기본 후보
    rect = cv2.minAreaRect(hull)
    quads.append(robust_order_points(cv2.boxPoints(rect).astype("float32")))
    
    # 2. 다각형 근사 및 3점 기준 단일 모서리 복원 후보(4방향)
    for eps in [0.02, 0.03, 0.04, 0.015, 0.05]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ordered = robust_order_points(approx.reshape(4, 2).astype("float32"))
            quads.append(ordered)
            
            orig_area = cv2.contourArea(ordered)
            # 4개 모서리 중 1개가 빛 반사/그림자로 밖으로 끌려나간 경우를 대비해
            # 나머지 온전한 3개 모서리로 직사각형 대칭점을 복원한 후보들을 함께 생성
            for k in range(4):
                opp = (k + 2) % 4
                fixed = ordered.copy()
                fixed[k] = ordered[(k - 1) % 4] + ordered[(k + 1) % 4] - ordered[opp]
                if cv2.isContourConvex(fixed):
                    fixed_area = cv2.contourArea(fixed)
                    # 바깥으로 삐져나간 돌출부를 안쪽으로 당겨 복원하는 경우(면적 70%~105%)만 유효 후보로 추가
                    if orig_area * 0.70 <= fixed_area <= orig_area * 1.05:
                        quads.append(robust_order_points(fixed))
            break
            
    return quads

def intersect_lines(a_x, b_x, a_y, b_y):
    """세로선(x = a_x*y + b_x)과 가로선(y = a_y*x + b_y)의 교점 계산"""
    denom = 1.0 - a_x * a_y
    if abs(denom) < 1e-5:
        return b_x, b_y
    x = (a_x * b_y + b_x) / denom
    y = a_y * x + b_y
    return x, y

def safe_snap_outer_wedge(warped, target_w=350, target_h=450):
    """
    인물(얼굴/머리카락) 영역은 절대 건드리지 않고,
    가장자리에 끼어든 책상/배경 삼각형 여백만 안전하게 감지하여 제거
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype("float32")
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype("float32")
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype("float32")
    
    # 얼굴 바깥쪽 안전한 스튜디오 배경 구역(좌/우 관자놀이 옆 여백)에서 기준 배경색 측정
    left_anchor = warped[45:135, 68:88]
    right_anchor = warped[45:135, target_w-88:target_w-68]
    
    # 두 앵커 중 더 밝고 채도가 낮은(깨끗한 인화지 배경) 쪽을 기준점으로 선택
    l_score = float(cv2.cvtColor(left_anchor, cv2.COLOR_BGR2GRAY).mean()) - float(cv2.cvtColor(left_anchor, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    r_score = float(cv2.cvtColor(right_anchor, cv2.COLOR_BGR2GRAY).mean()) - float(cv2.cvtColor(right_anchor, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    
    ref_patch = left_anchor if l_score >= r_score else right_anchor
    ref_lab = np.median(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
    ref_gray = float(np.median(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY)))
    ref_sat = float(np.median(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2HSV)[:, :, 1]))
    
    # 각 픽셀이 인화지 배경(또는 흰 테두리)인지 판별하는 마스크
    lab_dist = np.linalg.norm(lab - ref_lab, axis=2)
    is_studio_bg = (lab_dist < 28.0) & (gray > max(125.0, ref_gray - 42.0)) & (hsv[:, :, 1] < max(42.0, ref_sat + 25.0))
    is_white_border = (gray > 175.0) & (hsv[:, :, 1] < 30.0)
    is_card = is_studio_bg | is_white_border
    
    # 1. 좌측 외곽 여백 스캔 (x=78 안전 지점에서 바깥쪽 x=0으로만 이동)
    left_pts = []
    for y in range(35, int(target_h * 0.65), 4):
        start_x = -1
        for x in range(82, 35, -1):
            if is_card[y, x]:
                start_x = x
                break
        if start_x != -1:
            edge_x = 0
            for x in range(start_x, 1, -1):
                if not is_card[y, x] and not is_card[y, max(0, x - 1)]:
                    edge_x = x + 1
                    break
            left_pts.append([edge_x, y])
            
    # 2. 우측 외곽 여백 스캔 (x=target_w-78 안전 지점에서 바깥쪽 x=target_w-1로만 이동)
    right_pts = []
    for y in range(35, int(target_h * 0.65), 4):
        start_x = -1
        for x in range(target_w - 82, target_w - 35):
            if is_card[y, x]:
                start_x = x
                break
        if start_x != -1:
            edge_x = target_w - 1
            for x in range(start_x, target_w - 2):
                if not is_card[y, x] and not is_card[y, min(target_w - 1, x + 1)]:
                    edge_x = x - 1
                    break
            right_pts.append([edge_x, y])
            
    # 3. 상단 외곽 여백 스캔 (정수리 위쪽 y=38 안전 지점에서 위쪽 y=0으로만 이동)
    top_pts = []
    for x in range(65, target_w - 65, 4):
        start_y = -1
        for y in range(38, 12, -1):
            if is_card[y, x]:
                start_y = y
                break
        if start_y != -1:
            edge_y = 0
            for y in range(start_y, 1, -1):
                if not is_card[y, x] and not is_card[max(0, y - 1), x]:
                    edge_y = y + 1
                    break
            top_pts.append([x, edge_y])
            
    # 직선 피팅 (x = a*y + b 또는 y = a*x + b)
    a_L, b_L = 0.0, 0.0
    if len(left_pts) >= 10:
        arr = np.array(left_pts, dtype="float32")
        if np.median(arr[:, 0]) > 3.0 or np.percentile(arr[:, 0], 80) > 10.0:
            vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
            if abs(vy) > 1e-4 and abs(vx / vy) < 0.35:
                a_L = float(vx / vy)
                b_L = float(x0 - a_L * y0)
                
    a_R, b_R = 0.0, float(target_w - 1)
    if len(right_pts) >= 10:
        arr = np.array(right_pts, dtype="float32")
        if np.median(arr[:, 0]) < target_w - 4.0 or np.percentile(arr[:, 0], 20) < target_w - 11.0:
            vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
            if abs(vy) > 1e-4 and abs(vx / vy) < 0.35:
                a_R = float(vx / vy)
                b_R = float(x0 - a_R * y0)
                
    a_T, b_T = 0.0, 0.0
    if len(top_pts) >= 10:
        arr = np.array(top_pts, dtype="float32")
        if np.median(arr[:, 1]) > 3.0 or np.percentile(arr[:, 1], 80) > 8.0:
            vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
            if abs(vx) > 1e-4 and abs(vy / vx) < 0.35:
                a_T = float(vy / vx)
                b_T = float(y0 - a_T * x0)
                
    tl_x, tl_y = intersect_lines(a_L, b_L, a_T, b_T)
    tr_x, tr_y = intersect_lines(a_R, b_R, a_T, b_T)
    bl_x, bl_y = intersect_lines(a_L, b_L, 0.0, float(target_h - 1))
    br_x, br_y = intersect_lines(a_R, b_R, 0.0, float(target_h - 1))
    
    # 안전 범위 제한: 절대로 얼굴/어깨 안쪽으로 파고들지 못하도록 최대 보정 폭 제한
    tl_x = float(np.clip(tl_x, 0.0, target_w * 0.22))
    tl_y = float(np.clip(tl_y, 0.0, target_h * 0.12))
    tr_x = float(np.clip(tr_x, target_w * 0.78, target_w - 1.0))
    tr_y = float(np.clip(tr_y, 0.0, target_h * 0.12))
    bl_x = float(np.clip(bl_x, 0.0, target_w * 0.18))
    br_x = float(np.clip(br_x, target_w * 0.82, target_w - 1.0))
    
    # 보정량이 미세하면 원본 그대로 반환
    if tl_x < 3.0 and tl_y < 3.0 and (target_w - 1 - tr_x) < 3.0 and tr_y < 3.0:
        return warped
        
    src_quad = np.array([
        [tl_x, tl_y],
        [tr_x, tr_y],
        [br_x, float(target_h - 1)],
        [bl_x, float(target_h - 1)]
    ], dtype="float32")
    
    dst_quad = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    return cv2.warpPerspective(warped, M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def evaluate_id_photo_framing(img):
    """
    보정된 사진이 '정상적인 증명사진 구도'인지 정밀 채점:
    1) 정수리 위쪽 여백(헤드룸)이 살아있는가? (머리카락이 천장에 잘리면 대폭 감점)
    2) 좌상단/우상단 외곽 여백에 책상 바닥 침범 없이 밝고 대칭적인가?
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    
    # [핵심 1] 상단 중앙 정수리 위 여백(Headroom) 검사 (y: 5~28, x: 115~235)
    # 얼굴만 확대되어 검은 머리카락이 사진 맨 위를 덮어버린 경우를 원천 차단
    top_center_gray = gray[5:28, 115:235]
    headroom_bright = float(np.percentile(top_center_gray, 75))
    headroom_penalty = 0.0
    if headroom_bright < 115.0:
        headroom_penalty = 400.0  # 정수리가 잘린 과도한 확대 사진 강력 감점
        
    # [핵심 2] 좌측 상단 외곽 여백(x: 5~45, y: 15~200) vs 우측 상단 외곽 여백(x: 305~345, y: 15~200)
    # 책상 바닥이나 그림자 삼각형이 끼어들면 밝기가 떨어지고 채도/표준편차가 치솟음
    l_strip_g = gray[15:200, 5:45]
    r_strip_g = gray[15:200, w-45:w-5]
    l_strip_s = hsv[15:200, 5:45, 1]
    r_strip_s = hsv[15:200, w-45:w-5, 1]
    
    # 좌/우 여백 중 더 나쁜 쪽(책상이 침범한 쪽)의 상태 반영
    min_margin_bright = min(float(l_strip_g.mean()), float(r_strip_g.mean()))
    max_margin_sat = max(float(l_strip_s.mean()), float(r_strip_s.mean()))
    max_margin_std = max(float(l_strip_g.std()), float(r_strip_g.std()))
    lr_diff = abs(float(l_strip_g.mean()) - float(r_strip_g.mean())) + abs(float(l_strip_s.mean()) - float(r_strip_s.mean()))
    
    score = (
        min_margin_bright * 1.2
        - max_margin_sat * 1.5
        - max_margin_std * 1.2
        - lr_diff * 1.5
        - headroom_penalty
    )
    return score

def refine_and_deskew(warped, target_w=350, target_h=450):
    """외곽 여백 정밀 보정(품질 향상 시에만 채택) 및 미세 기울기 수평 보정"""
    if warped.shape[0] != target_h or warped.shape[1] != target_w:
        warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    # 외곽 삼각형 여백 스냅 시도 후, 품질 점수가 실제로 개선된 경우에만 채택
    snapped = safe_snap_outer_wedge(warped, target_w, target_h)
    if evaluate_id_photo_framing(snapped) > evaluate_id_photo_framing(warped) + 5.0:
        warped = snapped
        
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
                if abs(deg) < 10:
                    angles.append(deg)
                elif abs(deg - 180) < 10:
                    angles.append(deg - 180)
                elif abs(deg + 180) < 10:
                    angles.append(deg + 180)
                elif abs(abs(deg) - 90) < 10:
                    angles.append(deg - 90 if deg > 0 else deg + 90)
                
    if len(angles) >= 2:
        tilt = float(np.median(angles))
        if 0.25 <= abs(tilt) < 4.5:
            rot_M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt, 1.0)
            warped = cv2.warpAffine(warped, rot_M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    trim = 2
    final = warped[trim:target_h-trim, trim:target_w-trim]
    return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def is_actual_portrait_photo(cropped_img):
    """엄격한 증명사진 최종 자가 검증 (피부톤 + 이목구비 + 명암)"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    overall_mean = gray.mean()
    overall_std = gray.std()
    if overall_mean < 65 or overall_mean > 240 or overall_std < 22:
        return False
        
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
        
    gray_roi = gray[int(h * 0.15):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_count = (edges > 0).sum()
    if edge_count < 100:
        return False
        
    return True

def extract_all_photos(image):
    """최외곽 인화지 우선 탐색 및 3점 대칭 복원 보정 엔진"""
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
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 나무 책상(고채도)과 인화지(저채도·고명도) 분리 마스크 + 다중 이진화
    paper_hsv_mask = cv2.inRange(hsv_full, np.array([0, 0, 140], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8))
    _, thresh_bright = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    edged = cv2.Canny(blur, 35, 120)
    
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_e)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    closed_paper = cv2.morphologyEx(paper_hsv_mask, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    
    all_cnts = []
    for m_img in [closed_paper, closed_bright, closed_otsu]:
        cnts, _ = cv2.findContours(m_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)
    for m_img in [closed_adapt, edged_closed]:
        cnts, _ = cv2.findContours(m_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)
        
    raw_contours = []
    for c in all_cnts:
        area = cv2.contourArea(c)
        if not (img_area * 0.006 < area < img_area * 0.45):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        # 증명사진 및 폴라로이드 비율 필터 (빛 반사가 길게 합쳐진 0.53 비율 등 배제)
        if not (0.62 < ar < 0.95):
            continue
            
        raw_contours.append({
            'cnt': c,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area
        })
        
    if not raw_contours:
        return []
        
    # 면적이 큰 순서대로 정렬 (내부 얼굴/어깨 조각이 외곽 인화지를 이기지 못하도록 보장)
    raw_contours.sort(key=lambda x: x['area'], reverse=True)
    
    # 위치별 클러스터링
    clusters = []
    for item in raw_contours:
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
            
    # 좌상단부터 순서대로 클러스터 정렬
    clusters.sort(key=lambda cl: (cl[0]['center'][1] // 200, cl[0]['center'][0]))
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    verified_photos = []
    for cl in clusters:
        max_area_in_cluster = max(x['area'] for x in cl)
        # [핵심] 최외곽 인화지 면적의 72% 미만인 내부 윤곽선(머리/얼굴/어깨 조각)은 완전 배제!
        outer_items = [x for x in cl if x['area'] >= max_area_in_cluster * 0.72]
        
        best_photo = None
        best_q_score = -1e9
        
        for item in outer_items[:4]:
            quad_candidates = get_true_quads(item['cnt'])
            for q_pts in quad_candidates:
                ordered_pts = robust_order_points(q_pts)
                M = cv2.getPerspectiveTransform(ordered_pts, dst)
                warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
                refined = refine_and_deskew(warped, target_w, target_h)
                
                if is_actual_portrait_photo(refined):
                    q_score = evaluate_id_photo_framing(refined)
                    if q_score > best_q_score:
                        best_q_score = q_score
                        best_photo = refined
                        
        if best_photo is not None:
            verified_photos.append(best_photo)
            
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

# 1차 스캔 및 제출 버튼
if st.button("사진 제출 및 자동 스캔 완료", type="primary"):
    st.session_state["candidates"] = None
    st.session_state["upload_result"] = None
    
    if not target_gas_url:
        st.error("🚨 담임 선생님의 구글 드라이브 연결 정보가 없습니다! 담임 선생님께서 공유해주신 전용 링크로 다시 접속해주세요.")
    elif not student_id or not student_name:
        st.warning("⚠️ 학번과 이름을 모두 입력해주세요!")
    elif not uploaded_file:
        st.warning("⚠️ 증명사진 파일을 업로드해주세요!")
    else:
        with st.spinner("사진을 분석하고 모서리 4점을 정밀 보정 중입니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                processed_photos = extract_all_photos(image)
                
                # [경우 1] 증명사진이 감지되지 않은 경우 -> 전송 차단 및 재촬영 안내
                if not processed_photos:
                    st.error("🚨 **사진 속에서 규격에 맞는 증명사진을 감지하지 못했습니다!**\n\n- 빛 반사가 심하거나 증명사진이 너무 작게 찍혔습니다.\n- 사진이 잘 보이도록 조금 더 가까이 촬영한 후 다시 업로드해 주세요.")
                
                # [경우 2] 증명사진이 딱 1장 감지된 경우 -> 즉시 업로드 완료
                elif len(processed_photos) == 1:
                    photo = processed_photos[0]
                    is_success, buffer = cv2.imencode(".jpg", photo)
                    if is_success:
                        f_name = f"{student_id.strip()}_{student_name.strip()}.jpg"
                        upload_via_gas(buffer.tobytes(), f_name, target_gas_url)
                        st.session_state["upload_result"] = (f_name, photo)
                    else:
                        st.error("이미지 변환 중 오류가 발생했습니다.")
                
                # [경우 3] 사진이 2장 이상 감지된 경우 -> 학생 선택 모드 활성화
                else:
                    st.session_state["candidates"] = processed_photos
                    st.session_state["student_meta"] = (student_id.strip(), student_name.strip())
                    
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")

# 다중 사진 감지 시: 학생이 제출할 사진 1장을 선택하는 UI
if st.session_state["candidates"] is not None:
    cands = st.session_state["candidates"]
    s_id, s_name = st.session_state["student_meta"]
    
    st.warning(f"🧐 **사진 속에서 총 {len(cands)}장의 인물 사진이 감지되었습니다!**\n\n아래 사진 중 선생님께 제출할 **본인의 증명사진 1장**을 선택해 주세요.")
    
    cols = st.columns(len(cands))
    for idx, (col, p_img) in enumerate(zip(cols, cands)):
        with col:
            st.image(p_img, channels="BGR", caption=f"감지된 사진 {idx + 1}")
            if st.button(f"✅ {idx + 1}번 사진으로 제출", key=f"sel_btn_{idx}", type="primary", use_container_width=True):
                with st.spinner("선택한 증명사진을 선생님 클라우드에 저장 중입니다..."):
                    try:
                        is_success, buffer = cv2.imencode(".jpg", p_img)
                        if is_success:
                            f_name = f"{s_id}_{s_name}.jpg"
                            upload_via_gas(buffer.tobytes(), f_name, target_gas_url)
                            st.session_state["upload_result"] = (f_name, p_img)
                            st.session_state["candidates"] = None
                            st.rerun()
                    except Exception as e:
                        st.error(f"전송 중 오류가 발생했습니다: {e}")

# 최종 업로드 완료 화면 표시
if st.session_state["upload_result"] is not None:
    f_name, final_img = st.session_state["upload_result"]
    st.success(f"🎉 **[{f_name}]** 증명사진이 성공적으로 제출 및 자동 보정되었습니다!")
    st.image(final_img, channels="BGR", caption=f"선생님 드라이브에 저장된 최종 보정 사진 ({f_name})", width=280)

# 학생 화면 하단 관리자 복귀 링크
if current_code and not st.session_state.get("is_teacher_creator", False):
    st.markdown("<br><hr><center><small style='color: gray;'>선생님이신가요? <a href='.' style='color: gray; text-decoration: underline;'>새 링크 만들기</a></small></center>", unsafe_allow_html=True)
