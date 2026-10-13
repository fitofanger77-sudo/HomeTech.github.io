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
    """45도 이상 기울어진 사진도 짧은 변(위/아래)과 긴 변(좌/우)을 구분하여 [좌상, 우상, 우하, 좌하]로 정확히 정렬"""
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

def regularize_quad(pts):
    """형광등 빛 반사 등으로 1개 모서리가 바깥으로 튀었을 때, 반대쪽 직각(90도) 모서리 3개를 기준으로 직사각형 복원"""
    ordered = robust_order_points(pts)
    
    # 4개 꼭짓점의 내각 및 변의 길이 계산
    angles = []
    edges = []
    for i in range(4):
        p_curr = ordered[i]
        p_prev = ordered[(i - 1) % 4]
        p_next = ordered[(i + 1) % 4]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        edges.append(n2)
        if n1 == 0 or n2 == 0:
            angles.append(0.0)
        else:
            cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_th)))
            
    w_top, h_right, w_bot, h_left = edges[0], edges[1], edges[2], edges[3]
    w_diff = abs(w_top - w_bot) / max(w_top, w_bot, 1.0)
    h_diff = abs(h_left - h_right) / max(h_left, h_right, 1.0)
    
    # 마주 보는 변의 길이가 8% 이상 차이 나면 한쪽 모서리가 빛 반사/그림자로 왜곡된 상태
    if w_diff > 0.08 or h_diff > 0.08:
        # 각 꼭짓점의 인접 두 변 길이 합 비교 (튀어나간 꼭짓점은 인접 변의 합이 비정상적으로 큼)
        adj_sums = [edges[i] + edges[(i - 1) % 4] for i in range(4)]
        for k in range(4):
            opp = (k + 2) % 4
            # 반대쪽 모서리가 직각(82~98도)에 가깝고, 현재 모서리 k가 비정상적으로 길게 뻗어 나간 경우
            if 82.0 <= angles[opp] <= 98.0 and adj_sums[k] > adj_sums[opp] * 1.08:
                fixed = ordered.copy()
                fixed[k] = ordered[(k + 1) % 4] + ordered[(k + 3) % 4] - ordered[opp]
                return robust_order_points(fixed)
                
    return ordered

def get_true_quads(contour):
    """인화지의 4개 모서리 후보(원본 다각형 + 직사각형 복원본 + 최소외접사각형) 추출"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    quads = []
    
    for eps in [0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            raw_q = approx.reshape(4, 2).astype("float32")
            quads.append(regularize_quad(raw_q))
            break
            
    rect = cv2.minAreaRect(hull)
    quads.append(robust_order_points(cv2.boxPoints(rect).astype("float32")))
    return quads

def fit_border_line_x_of_y(pts_list, default_x):
    """좌/우 세로 경계 점들로부터 x = a*y + b 직선 피팅 (이상치 제거)"""
    if len(pts_list) < 12:
        return 0.0, float(default_x)
    pts_arr = np.array(pts_list, dtype="float32")
    vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-4:
        return 0.0, float(default_x)
    a = float(vx / vy)
    b = float(x0 - a * y0)
    # 기울기가 25도 이상 극단적으로 꺾인 직선은 무시
    if abs(a) > 0.45:
        return 0.0, float(np.median(pts_arr[:, 0]))
    return a, b

def fit_border_line_y_of_x(pts_list, default_y):
    """상/하 가로 경계 점들로부터 y = a*x + b 직선 피팅 (이상치 제거)"""
    if len(pts_list) < 12:
        return 0.0, float(default_y)
    pts_arr = np.array(pts_list, dtype="float32")
    vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
    if abs(vx) < 1e-4:
        return 0.0, float(default_y)
    a = float(vy / vx)
    b = float(y0 - a * x0)
    if abs(a) > 0.45:
        return 0.0, float(np.median(pts_arr[:, 1]))
    return a, b

def intersect_lines(a_x, b_x, a_y, b_y):
    """세로선(x = a_x*y + b_x)과 가로선(y = a_y*x + b_y)의 교점 계산"""
    denom = 1.0 - a_x * a_y
    if abs(denom) < 1e-5:
        return b_x, b_y
    x = (a_x * b_y + b_x) / denom
    y = a_y * x + b_y
    return x, y

def refine_quad_in_warped(warped, target_w=350, target_h=450):
    """1차 스캔된 이미지 내부에서 실제 인화지 4변을 다시 찾아 빈틈없이 꽉 채우도록 재투영"""
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype("float32")
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype("float32")
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype("float32")
    
    # 사진 내부의 밝은 배경(인화지 여백/스튜디오 배경) 기준 색상 측정
    # 상단 좌/우 배경 영역에서 밝고 채도가 낮은 픽셀 샘플링
    bg_zone_gray = np.concatenate([
        gray[35:180, 45:105].flatten(),
        gray[35:180, target_w-105:target_w-45].flatten()
    ])
    bg_thresh = np.percentile(bg_zone_gray, 70)
    
    # 인화지/스튜디오 배경 판별 기준: 충분히 밝고 나무 책상처럼 채도가 높지 않은 영역
    is_paper = (gray >= max(135.0, bg_thresh - 35.0)) & (hsv[:, :, 1] < 55.0)
    
    # 1. 좌측 경계 탐색 (안쪽 배경 x=95 에서 바깥쪽 x=0 방향으로 스캔)
    left_pts = []
    for y in range(45, int(target_h * 0.72), 3):
        # 안쪽에서 인화지 배경 시작점 찾기
        start_x = -1
        for x in range(105, 15, -1):
            if is_paper[y, x]:
                start_x = x
                break
        if start_x != -1:
            edge_x = 0
            ref_l, ref_a, ref_b = lab[y, start_x]
            for x in range(start_x, 1, -1):
                # 바깥쪽 픽셀과의 색상/밝기 차이 계산
                dl = abs(lab[y, x - 1, 0] - ref_l)
                da = abs(lab[y, x - 1, 1] - ref_a)
                db = abs(lab[y, x - 1, 2] - ref_b)
                color_dist = np.sqrt(dl*dl + da*da + db*db)
                if (color_dist > 18.0 and not is_paper[y, x - 1]) or gray[y, x - 1] < 115.0:
                    edge_x = x
                    break
            left_pts.append([edge_x, y])
            
    # 2. 우측 경계 탐색 (안쪽 배경에서 바깥쪽 x=target_w-1 방향으로 스캔)
    right_pts = []
    for y in range(45, int(target_h * 0.72), 3):
        start_x = -1
        for x in range(target_w - 105, target_w - 15):
            if is_paper[y, x]:
                start_x = x
                break
        if start_x != -1:
            edge_x = target_w - 1
            ref_l, ref_a, ref_b = lab[y, start_x]
            for x in range(start_x, target_w - 2):
                dl = abs(lab[y, x + 1, 0] - ref_l)
                da = abs(lab[y, x + 1, 1] - ref_a)
                db = abs(lab[y, x + 1, 2] - ref_b)
                color_dist = np.sqrt(dl*dl + da*da + db*db)
                if (color_dist > 18.0 and not is_paper[y, x + 1]) or gray[y, x + 1] < 115.0:
                    edge_x = x
                    break
            right_pts.append([edge_x, y])
            
    # 3. 상단 경계 탐색 (안쪽 y=80 에서 위쪽 y=0 방향으로 스캔)
    top_pts = []
    for x in range(45, target_w - 45, 3):
        start_y = -1
        for y in range(85, 10, -1):
            if is_paper[y, x]:
                start_y = y
                break
        if start_y != -1:
            edge_y = 0
            ref_l, ref_a, ref_b = lab[start_y, x]
            for y in range(start_y, 1, -1):
                dl = abs(lab[y - 1, x, 0] - ref_l)
                da = abs(lab[y - 1, x, 1] - ref_a)
                db = abs(lab[y - 1, x, 2] - ref_b)
                color_dist = np.sqrt(dl*dl + da*da + db*db)
                if (color_dist > 18.0 and not is_paper[y - 1, x]) or gray[y - 1, x] < 115.0:
                    edge_y = y
                    break
            top_pts.append([x, edge_y])
            
    # 4. 하단 경계 탐색 (하단 외곽 여백/경계 스캔)
    bot_pts = []
    for x in range(35, target_w - 35, 3):
        edge_y = target_h - 1
        # 하단 끝에서 위로 올라가며 어두운 갈색 책상/서랍 바닥(채도 높거나 중간 밝기)이 끝나는 지점 확인
        for y in range(target_h - 1, target_h - 45, -1):
            # 책상/서랍 바닥 특징: 인화지(흰색)도 아니고 검은 정장(매우 어두움, gray < 55)도 아닌 배경
            is_bg_wedge = (60.0 < gray[y, x] < 155.0) and (hsv[y, x, 1] > 25.0)
            if is_bg_wedge:
                edge_y = y - 1
            else:
                break
        bot_pts.append([x, edge_y])
        
    a_L, b_L = fit_border_line_x_of_y(left_pts, 0)
    a_R, b_R = fit_border_line_x_of_y(right_pts, target_w - 1)
    a_T, b_T = fit_border_line_y_of_x(top_pts, 0)
    a_B, b_B = fit_border_line_y_of_x(bot_pts, target_h - 1)
    
    tl_x, tl_y = intersect_lines(a_L, b_L, a_T, b_T)
    tr_x, tr_y = intersect_lines(a_R, b_R, a_T, b_T)
    br_x, br_y = intersect_lines(a_R, b_R, a_B, b_B)
    bl_x, bl_y = intersect_lines(a_L, b_L, a_B, b_B)
    
    # 과도한 축소 방지 (최대 30% 이내 범위로 안전 클램핑)
    tl_x = np.clip(tl_x, 0, target_w * 0.30)
    tl_y = np.clip(tl_y, 0, target_h * 0.25)
    tr_x = np.clip(tr_x, target_w * 0.70, target_w - 1)
    tr_y = np.clip(tr_y, 0, target_h * 0.25)
    br_x = np.clip(br_x, target_w * 0.70, target_w - 1)
    br_y = np.clip(br_y, target_h * 0.75, target_h - 1)
    bl_x = np.clip(bl_x, 0, target_w * 0.30)
    bl_y = np.clip(bl_y, target_h * 0.75, target_h - 1)
    
    src_quad = np.array([
        [tl_x, tl_y],
        [tr_x, tr_y],
        [br_x, br_y],
        [bl_x, bl_y]
    ], dtype="float32")
    
    dst_quad = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    return cv2.warpPerspective(warped, M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def refine_and_deskew(warped, target_w=350, target_h=450):
    """2단계 자가 점검: 내부 4변 재투영(2회 반복) 후 미세 수평 보정"""
    if warped.shape[0] != target_h or warped.shape[1] != target_w:
        warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    # [핵심] 사다리꼴 여백 및 책상/서랍 침범 영역을 2회 반복 추적하여 캔버스에 100% 꽉 채움
    for _ in range(2):
        warped = refine_quad_in_warped(warped, target_w, target_h)
        
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
        if 0.25 <= abs(tilt) < 5.0:
            rot_M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt, 1.0)
            warped = cv2.warpAffine(warped, rot_M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    trim = 3
    final = warped[trim:target_h-trim, trim:target_w-trim]
    return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def score_photo_quality(cropped_img):
    """보정된 이미지가 얼마나 빈틈없이 반듯한 증명사진인지 품질 점수 계산"""
    h, w = cropped_img.shape[:2]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # 좌상단과 우상단 모서리의 배경 대칭성 (책상 침범이 없을수록 좌우 모서리 밝기/채도가 일치함)
    tl_patch = gray[5:35, 5:35]
    tr_patch = gray[5:35, w-35:w-5]
    symmetry_penalty = abs(float(tl_patch.mean()) - float(tr_patch.mean()))
    
    # 상단 모서리 평균 밝기 (증명사진 배경은 밝고 채도가 낮음)
    corner_bright = (float(tl_patch.mean()) + float(tr_patch.mean())) / 2.0
    corner_sat = (float(hsv[5:35, 5:35, 1].mean()) + float(hsv[5:35, w-35:w-5, 1].mean())) / 2.0
    
    return corner_bright - symmetry_penalty * 1.5 - corner_sat * 0.8

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
    """다단계 적응형 탐색 및 4변 직선 피팅 정밀 보정 엔진"""
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
    
    # 나무 책상(고채도)과 흰색 인화지(저채도·고명도)를 완벽 분리하는 전용 마스크 추가
    paper_hsv_mask = cv2.inRange(hsv_full, np.array([0, 0, 145], dtype=np.uint8), np.array([180, 52, 255], dtype=np.uint8))
    _, thresh_bright = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY)
    _, thresh_high = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    edged = cv2.Canny(blur, 35, 120)
    
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_e)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    closed_paper = cv2.morphologyEx(paper_hsv_mask, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_high = cv2.morphologyEx(thresh_high, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    closed_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    
    all_cnts = []
    for m_img in [closed_paper, closed_bright, closed_high, closed_otsu]:
        cnts, _ = cv2.findContours(m_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)
    for m_img in [closed_adapt, edged_closed]:
        cnts, _ = cv2.findContours(m_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts.extend(cnts)
        
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
            
        quad_list = get_true_quads(c)
        peri = cv2.arcLength(cv2.convexHull(c), True)
        approx = cv2.approxPolyDP(cv2.convexHull(c), 0.03 * peri, True)
        is_quad = 1.0 if len(approx) == 4 else 0.0
        
        ratio_diff = min(abs(ar - 0.75), abs(ar - 0.778))
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.15)) * 50.0
        base_score = is_quad * 50.0 + ratio_score + (area / img_area) * 80.0
        
        for q_pts in quad_list:
            candidates.append({
                'pts': q_pts,
                'center': (cx, cy),
                'w': w, 'h': h,
                'area': area,
                'score': base_score
            })
        
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 동일 위치의 사진 후보들을 그룹핑하여 가장 테두리가 완벽한 후보 선택
    clusters = []
    for cand in candidates:
        cx1, cy1 = cand['center']
        placed = False
        for cluster in clusters:
            cx2, cy2 = cluster[0]['center']
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < min(cand['w'], cluster[0]['w']) * 0.65:
                if len(cluster) < 4:
                    cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])
            
    # 좌상단부터 순서대로 클러스터 정렬
    clusters.sort(key=lambda cl: (cl[0]['center'][1] // 200, cl[0]['center'][0]))
    
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    verified_photos = []
    for cl in clusters:
        best_photo = None
        best_q_score = -1e9
        
        for p in cl:
            ordered_pts = robust_order_points(p['pts'])
            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
            refined = refine_and_deskew(warped, target_w, target_h)
            
            if is_actual_portrait_photo(refined):
                q_score = score_photo_quality(refined)
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
