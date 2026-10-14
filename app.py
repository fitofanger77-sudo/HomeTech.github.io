import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import os
import io
import base64
import requests
import urllib.parse

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

# 세션 상태 초기화
if "photo_groups" not in st.session_state:
    st.session_state["photo_groups"] = None
if "selected_photo_idx" not in st.session_state:
    st.session_state["selected_photo_idx"] = 0
if "selected_style_idx" not in st.session_state:
    st.session_state["selected_style_idx"] = 0
if "rotation_deg" not in st.session_state:
    st.session_state["rotation_deg"] = 0
if "student_meta" not in st.session_state:
    st.session_state["student_meta"] = ("", "")
if "final_uploaded" not in st.session_state:
    st.session_state["final_uploaded"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# 선생님 링크 및 QR 정보 세션 저장
if "share_url" not in st.session_state:
    st.session_state["share_url"] = ""
if "qr_api_url" not in st.session_state:
    st.session_state["qr_api_url"] = ""
if "qr_bytes" not in st.session_state:
    st.session_state["qr_bytes"] = None

def reset_scan_state():
    """재업로드/재촬영을 위해 모든 스캔 상태를 초기화"""
    st.session_state["photo_groups"] = None
    st.session_state["selected_photo_idx"] = 0
    st.session_state["selected_style_idx"] = 0
    st.session_state["rotation_deg"] = 0
    st.session_state["final_uploaded"] = None
    st.session_state["uploader_key"] += 1

def get_current_base_url():
    """현재 Streamlit 앱의 도메인(Base URL) 감지"""
    try:
        ctx = getattr(st, "context", None)
        if ctx and hasattr(ctx, "headers"):
            host = ctx.headers.get("host", "")
            proto = ctx.headers.get("x-forwarded-proto", "https")
            if host:
                return f"{proto}://{host}"
    except Exception:
        pass
    return ""

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
st.write("학번과 이름을 입력하고 증명사진을 올리면, 여러 형태의 스캔본 중 가장 마음에 드는 사진을 직접 골라 선생님께 제출할 수 있습니다.")

# --- ⚙️ [선생님 전용 영역] 다이렉트 URL & QR 코드 배포 센터 ---
if st.session_state.get("is_teacher_creator", False) and current_code:
    with st.container():
        st.success("🎉 **우리 반 전용 접속 링크와 QR 코드가 생성되었습니다!**")
        
        # 다이렉트 URL 및 QR 이미지가 세션에 없으면 생성 (단축 URL 제거)
        if not st.session_state["share_url"] or not st.session_state["qr_api_url"]:
            detected_base = get_current_base_url() or "https://share.streamlit.io"
            direct_url = f"{detected_base}/?t={current_code}"
            qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=10&data={urllib.parse.quote(direct_url)}"
            
            st.session_state["share_url"] = direct_url
            st.session_state["qr_api_url"] = qr_url
            try:
                st.session_state["qr_bytes"] = requests.get(qr_url, timeout=5).content
            except Exception:
                st.session_state["qr_bytes"] = None

        s_url = st.session_state["share_url"]
        qr_url = st.session_state["qr_api_url"]
        qr_bytes = st.session_state["qr_bytes"]

        qr_col, link_col = st.columns([1, 1.4])
        with qr_col:
            st.image(qr_url, width=220, caption="교실 TV / 가정통신문용 QR 코드")
            
            copy_qr_html = f"""
            <div style="margin-top: 5px; margin-bottom: 8px;">
                <button id="copyQrBtn" onclick="copyQrImage()" style="
                    width: 220px;
                    padding: 8px 12px;
                    background-color: #ff4b4b;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                ">📋 QR 코드 이미지 복사</button>
            </div>
            <script>
            async function copyQrImage() {{
                const btn = document.getElementById('copyQrBtn');
                try {{
                    btn.innerText = '⏳ 복사 중...';
                    const response = await fetch('{qr_url}');
                    const blob = await response.blob();
                    await navigator.clipboard.write([
                        new ClipboardItem({{ [blob.type]: blob }})
                    ]);
                    btn.innerText = '✅ 복사 완료! (Ctrl+V로 붙여넣기)';
                    setTimeout(() => {{ btn.innerText = '📋 QR 코드 이미지 복사'; }}, 3000);
                }} catch (err) {{
                    btn.innerText = '⚠️ 복사 실패 (아래 다운로드 이용)';
                    setTimeout(() => {{ btn.innerText = '📋 QR 코드 이미지 복사'; }}, 3000);
                }}
            }}
            </script>
            """
            components.html(copy_qr_html, height=48)
            
            if qr_bytes:
                st.download_button(
                    label="💾 QR 코드 이미지 다운로드 (PNG)",
                    data=qr_bytes,
                    file_name="우리반_증명사진_제출_QR코드.png",
                    mime="image/png",
                    use_container_width=False
                )

        with link_col:
            st.markdown("### 🔗 학생 배포용 접속 링크")
            st.code(s_url, language=None)
            
            copy_url_html = f"""
            <div style="margin-bottom: 12px;">
                <button id="copyUrlBtn" onclick="copyTextUrl()" style="
                    padding: 7px 14px;
                    background-color: #2e7bcf;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                    cursor: pointer;
                ">📋 접속 링크 복사하기</button>
            </div>
            <script>
            function copyTextUrl() {{
                const btn = document.getElementById('copyUrlBtn');
                navigator.clipboard.writeText('{s_url}').then(() => {{
                    btn.innerText = '✅ 링크 복사 완료!';
                    setTimeout(() => {{ btn.innerText = '📋 접속 링크 복사하기'; }}, 2500);
                }});
            }}
            </script>
            """
            components.html(copy_url_html, height=42)

            st.markdown(
                f"**💡 배포 팁:**\n"
                f"- **학급 단톡방 / 알림장:** 파란색 `접속 링크 복사`를 눌러 학생들에게 공유해 주세요. 누르는 즉시 페이지로 바로 들어옵니다.\n"
                f"- **가정통신문 / 유인물:** 빨간색 `QR 코드 이미지 복사`를 누른 뒤 **한글(HWP)이나 PPT에서 `Ctrl + V`**만 누르면 QR 코드가 들어갑니다.\n"
                f"- **교실 칠판:** 이 화면의 QR 코드를 교실 TV나 프로젝터에 띄워두시면 학생들이 스마트폰으로 바로 찍어 제출합니다."
            )
            
            if st.button("👀 학생 제출 화면 확인하기 (안내창 닫기)", type="secondary"):
                st.session_state["is_teacher_creator"] = False
                st.rerun()

# 전용 링크가 없는 기본 주소로 접속했을 때만 선생님 설정창 표시
if not current_code:
    with st.expander("⚙️ [선생님 전용] 우리 반 전용 수집 링크 만들기", expanded=True):
        st.markdown("구매하신 선생님이신가요? 본인의 **구글 앱스크립트(GAS) 웹 앱 URL**을 입력하시면, **우리 반 학생들에게 배포할 다이렉트 접속 링크와 QR 코드**를 즉시 만들어 드립니다.")
        gas_input = st.text_input("구글 앱스크립트(GAS) 웹 앱 URL 입력", value=target_gas_url, type="password")
        
        detected_host = get_current_base_url()
        base_domain_input = st.text_input(
            "현재 웹 앱 기본 주소 (자동 감지됨)", 
            value=detected_host if detected_host else "https://share.streamlit.io",
            help="학생들이 접속할 기본 웹사이트 주소입니다. 브라우저 주소창의 도메인이 맞는지 확인해 주세요."
        )
        
        if st.button("우리 반 전용 링크 & QR 코드 발급하기", type="primary"):
            if not gas_input.strip():
                st.warning("⚠️ 구글 앱스크립트 URL을 입력해주세요!")
            else:
                encoded_code = base64.urlsafe_b64encode(gas_input.strip().encode("utf-8")).decode("utf-8")
                base_clean = base_domain_input.strip().rstrip('/')
                direct_url = f"{base_clean}/?t={encoded_code}"
                
                with st.spinner("다이렉트 접속 URL 및 QR 코드를 생성하고 있습니다..."):
                    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=10&data={urllib.parse.quote(direct_url)}"
                    
                    st.session_state["is_teacher_creator"] = True
                    st.session_state["share_url"] = direct_url
                    st.session_state["qr_api_url"] = qr_url
                    try:
                        st.session_state["qr_bytes"] = requests.get(qr_url, timeout=5).content
                    except Exception:
                        st.session_state["qr_bytes"] = None
                        
                    st.query_params["t"] = encoded_code
                    st.rerun()

st.markdown("---")

# --- 1단계: 학생 정보 및 업로드 방식 선택 ---
st.subheader("1️⃣ 학번·이름 입력 및 업로드 방식 선택")
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("학번 (예: 1학년 1반 1번인 경우, 10101)")
with col2:
    student_name = st.text_input("이름 (예: 홍길동)")

upload_mode = st.radio(
    "증명사진 파일 종류를 선택해 주세요:",
    options=[
        "📸 실물 증명사진 찍은 파일 올리기 (배경 자동 제거 · 여러 스캔본 중 선택)",
        "🖼️ 디지털 증명사진 원본 파일 올리기 (사진관 파일 등 · 스캔 없이 규격만 맞춤)"
    ],
    index=0
)

is_direct_digital_mode = ("디지털 증명사진 원본" in upload_mode)

if is_direct_digital_mode:
    st.caption("💡 사진관에서 받은 이미지 파일이나 이미 여백 없이 잘라진 증명사진 파일을 선택해 주세요.")
else:
    st.caption("💡 종이 증명사진을 책상 위에 놓고 찍은 사진 파일을 올려주세요. (스마트폰에서는 파일 선택 버튼을 누르면 기본 카메라로 바로 촬영할 수도 있습니다.)")

uploaded_file = st.file_uploader(
    "증명사진 파일 선택 (JPG, PNG)",
    type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
    key=f"uploader_{st.session_state['uploader_key']}"
)

# ==========================================================
# 영상 처리 및 다중 스캔 스타일 생성 알고리즘
# ==========================================================
def robust_order_points(pts):
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

def gentle_deskew(warped, target_w=350, target_h=450, trim_px=3):
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
            
    if trim_px > 0:
        final = warped[trim_px:target_h-trim_px, trim_px:target_w-trim_px]
        return cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return warped

def enhance_studio_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    l_blended = cv2.addWeighted(l_enhanced, 0.65, l_ch, 0.35, 6)
    
    merged = cv2.merge((l_blended, a_ch, b_ch))
    bright_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(bright_img, (0, 0), 1.2)
    sharpened = cv2.addWeighted(bright_img, 1.25, gaussian, -0.25, 0)
    return sharpened

def get_skin_mask(hsv_img):
    lower_skin1 = np.array([0, 22, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 195, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 22, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 195, 255], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lower_skin1, upper_skin1),
        cv2.inRange(hsv_img, lower_skin2, upper_skin2)
    )

def is_valid_portrait_candidate(cropped_img):
    h, w = cropped_img.shape[:2]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY).astype("float32")
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    skin_mask = get_skin_mask(hsv)
    
    overall_mean = float(gray.mean())
    overall_std = float(gray.std())
    if overall_mean < 55.0 or overall_mean > 242.0 or overall_std < 22.0:
        return False
        
    center_skin = skin_mask[int(h * 0.18):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    center_skin_ratio = float((center_skin > 0).mean())
    overall_skin_ratio = float((skin_mask > 0).mean())
    
    if center_skin_ratio < 0.07 or center_skin_ratio > 0.66 or overall_skin_ratio > 0.50:
        return False
        
    top_edge_skin = float((skin_mask[0:int(h * 0.08), :] > 0).mean())
    if top_edge_skin > 0.18:
        return False
        
    gray_roi = cv2.cvtColor(cropped_img[int(h * 0.15):int(h * 0.78), int(w * 0.18):int(w * 0.82)], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 45, 140)
    if (edges > 0).sum() < 95:
        return False
        
    return True

def score_rect_cleanliness(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    
    tl_g = gray[6:42, 6:42]
    tr_g = gray[6:42, w-42:w-6]
    tl_s = hsv[6:42, 6:42, 1]
    tr_s = hsv[6:42, w-42:w-6, 1]
    
    lr_diff = abs(float(tl_g.mean()) - float(tr_g.mean())) + abs(float(tl_s.mean()) - float(tr_s.mean()))
    min_bright = min(float(tl_g.mean()), float(tr_g.mean()))
    max_std = max(float(tl_g.std()), float(tr_g.std()))
    
    return min_bright - lr_diff * 1.6 - max_std * 1.2

def build_3_scan_styles_for_cluster(image, cluster_items):
    target_w, target_h = 350, 450
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    
    max_area = max(x['area'] for x in cluster_items)
    valid_items = [x for x in cluster_items if x['area'] >= max_area * 0.60]
    
    all_candidates = []
    for item in valid_items[:5]:
        hull = item['cnt']
        peri = cv2.arcLength(hull, True)
        
        rect = cv2.minAreaRect(hull)
        r_pts = robust_order_points(cv2.boxPoints(rect).astype("float32"))
        M_r = cv2.getPerspectiveTransform(r_pts, dst)
        w_r = gentle_deskew(cv2.warpPerspective(image, M_r, (target_w, target_h), flags=cv2.INTER_CUBIC), target_w, target_h, trim_px=3)
        if is_valid_portrait_candidate(w_r):
            all_candidates.append((score_rect_cleanliness(w_r), w_r))
            
        for eps in [0.02, 0.025, 0.03]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                ordered = robust_order_points(approx.reshape(4, 2).astype("float32"))
                M_p = cv2.getPerspectiveTransform(ordered, dst)
                w_p = gentle_deskew(cv2.warpPerspective(image, M_p, (target_w, target_h), flags=cv2.INTER_CUBIC), target_w, target_h, trim_px=3)
                if is_valid_portrait_candidate(w_p):
                    all_candidates.append((score_rect_cleanliness(w_p), w_p))
                break
                
    if not all_candidates:
        return None
        
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    style_a = all_candidates[0][1]
    
    tx_tight, ty_tight = int(target_w * 0.05), int(target_h * 0.05)
    tight_crop = style_a[ty_tight:target_h-ty_tight, tx_tight:target_w-tx_tight]
    style_b = cv2.resize(tight_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    style_c = enhance_studio_lighting(style_a)
    
    return [
        ("타입 A: 표준 자동 스캔 (추천)", style_a),
        ("타입 B: 테두리 밀착 스캔 (여백 제거)", style_b),
        ("타입 C: 화사한 조명·선명도 보정본", style_c)
    ]

def extract_photo_groups(image, is_digital_mode=False):
    target_w, target_h = 350, 450
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    top_b = image[0:5, :, :]
    bottom_b = image[img_h-5:img_h, :, :]
    left_b = image[:, 0:5, :]
    right_b = image[:, img_w-5:img_w, :]
    border_concat = np.concatenate([top_b.flatten(), bottom_b.flatten(), left_b.flatten(), right_b.flatten()])
    
    if is_digital_mode or np.mean(border_concat) > 155:
        target_ratio = target_w / float(target_h)
        cur_ratio = img_w / float(img_h)
        if cur_ratio > target_ratio:
            new_w = int(img_h * target_ratio)
            x0 = (img_w - new_w) // 2
            center_cropped = cv2.resize(image[:, x0:x0+new_w], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            new_h = int(img_w / target_ratio)
            y0 = (img_h - new_h) // 2
            center_cropped = cv2.resize(image[y0:y0+new_h, :], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
        base_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        tx, ty = int(target_w * 0.04), int(target_h * 0.04)
        margin_trimmed = cv2.resize(center_cropped[ty:target_h-ty, tx:target_w-tx], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        studio_bright = enhance_studio_lighting(center_cropped)
        
        if is_digital_mode or is_valid_portrait_candidate(base_resized):
            return [[
                ("타입 A: 원본 비율 유지 스캔 (추천)", center_cropped),
                ("타입 B: 테두리 밀착 스캔 (여백 제거)", margin_trimmed),
                ("타입 C: 화사한 조명·선명도 보정본", studio_bright)
            ]]
            
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edged = cv2.Canny(blur, 30, 140)
    combined_edge = cv2.bitwise_or(thresh_adapt, edged)
    
    paper_hsv = cv2.inRange(hsv_full, np.array([0, 0, 135], dtype=np.uint8), np.array([180, 55, 255], dtype=np.uint8))
    _, thresh_bright = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
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
        if not (img_area * 0.012 < area < img_area * 0.65):
            continue
        rect = cv2.minAreaRect(hull)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        if w > h:
            w, h = h, w
        ar = w / h
        if not (0.60 < ar < 0.95):
            continue
            
        raw_items.append({
            'cnt': hull,
            'center': (cx, cy),
            'w': w, 'h': h,
            'area': area
        })
        
    if not raw_items:
        return []
        
    raw_items.sort(key=lambda x: x['area'], reverse=True)
    
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
    
    photo_groups = []
    for cl in clusters:
        styles = build_3_scan_styles_for_cluster(image, cl)
        if styles is not None:
            photo_groups.append(styles)
            
    return photo_groups

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

# --- 1차 스캔 실행 버튼 ---
if st.button("🔍 사진 스캔 및 스캔본 확인하기", type="primary", use_container_width=True):
    st.session_state["photo_groups"] = None
    st.session_state["selected_photo_idx"] = 0
    st.session_state["selected_style_idx"] = 0
    st.session_state["rotation_deg"] = 0
    st.session_state["final_uploaded"] = None
    
    if not target_gas_url:
        st.error("🚨 담임 선생님의 구글 드라이브 연결 정보가 없습니다! 담임 선생님께서 공유해주신 전용 링크로 다시 접속해주세요.")
    elif not student_id.strip() or not student_name.strip():
        st.warning("⚠️ 학번과 이름을 모두 입력해주세요!")
    elif uploaded_file is None:
        st.warning("⚠️ 먼저 증명사진 파일을 업로드해주세요!")
    else:
        with st.spinner("사진을 분석하여 깔끔한 스캔본들을 생성하고 있습니다..."):
            try:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                groups = extract_photo_groups(image, is_digital_mode=is_direct_digital_mode)
                
                if not groups:
                    st.error(
                        "🚨 **사진 속에서 증명사진을 감지하지 못했습니다! 다시 촬영해 주세요.**\n\n"
                        "**[재촬영 안내]**\n"
                        "1. 증명사진이 너무 멀리 찍혔거나 형광등 빛 반사가 심할 수 있습니다.\n"
                        "2. 어두운 책상이나 마우스패드 위에 사진을 놓고 **네 모서리가 모두 보이도록** 다시 촬영해 주세요.\n"
                        "3. 만약 사진관에서 받은 원본 파일이라면 위에서 **'🖼️ 디지털 증명사진 원본 파일 올리기'**를 선택해 주세요."
                    )
                else:
                    st.session_state["photo_groups"] = groups
                    st.session_state["student_meta"] = (student_id.strip(), student_name.strip())
            except Exception as e:
                st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")

# --- 2단계: 다중 사진 선택 & 3가지 스캔 형태 선택 UI ---
if st.session_state["photo_groups"] is not None:
    groups = st.session_state["photo_groups"]
    s_id, s_name = st.session_state["student_meta"]
    
    # [A] 사진이 2장 이상 감지된 경우: 어떤 사진인지 먼저 선택
    if len(groups) > 1:
        st.markdown("---")
        st.subheader(f"2️⃣ 감지된 {len(groups)}장의 사진 중 내 사진 고르기")
        st.info(f"🧐 업로드한 이미지에서 총 **{len(groups)}장**의 인물 사진이 발견되었습니다! 먼저 제출할 본인의 사진을 클릭해 주세요.")
        
        p_cols = st.columns(len(groups))
        for p_idx, (p_col, style_list) in enumerate(zip(p_cols, groups)):
            with p_col:
                rep_img = style_list[0][1]
                is_picked_photo = (st.session_state["selected_photo_idx"] == p_idx)
                st.image(rep_img, channels="BGR", caption=f"감지된 사진 #{p_idx + 1}")
                btn_txt = f"✅ 사진 #{p_idx + 1} 선택됨" if is_picked_photo else f"사진 #{p_idx + 1} 선택하기"
                if st.button(btn_txt, key=f"pick_photo_{p_idx}", type="primary" if is_picked_photo else "secondary", use_container_width=True):
                    st.session_state["selected_photo_idx"] = p_idx
                    st.session_state["selected_style_idx"] = 0
                    st.session_state["rotation_deg"] = 0
                    st.rerun()

    # [B] 선택된 사진에 대한 3가지 스캔본 제시 및 선택
    st.markdown("---")
    step_num = "3️⃣" if len(groups) > 1 else "2️⃣"
    st.subheader(f"{step_num} 가장 마음에 드는 스캔 형태를 골라주세요")
    st.write("테두리 여백과 보정 방식에 따라 3가지 버전으로 스캔되었습니다. **가장 반듯하고 깔끔한 버전을 선택**해 주세요.")
    
    current_styles = groups[st.session_state["selected_photo_idx"]]
    s_cols = st.columns(len(current_styles))
    
    for s_idx, (s_col, (style_title, style_img)) in enumerate(zip(s_cols, current_styles)):
        with s_col:
            is_picked_style = (st.session_state["selected_style_idx"] == s_idx)
            st.image(style_img, channels="BGR", caption=style_title)
            s_btn_txt = "✔️ 이 스캔본 선택됨" if is_picked_style else "이 스캔본 선택"
            if st.button(s_btn_txt, key=f"pick_style_{s_idx}", type="primary" if is_picked_style else "secondary", use_container_width=True):
                st.session_state["selected_style_idx"] = s_idx
                st.rerun()

    # [C] 최종 제출 확인 및 재업로드 선택창
    st.markdown("---")
    final_step_num = "4️⃣" if len(groups) > 1 else "3️⃣"
    st.subheader(f"{final_step_num} 최종 확인 및 제출하기")
    
    chosen_title, chosen_img = current_styles[st.session_state["selected_style_idx"]]
    preview_img = chosen_img.copy()
    rot_k = (st.session_state["rotation_deg"] // 90) % 4
    if rot_k != 0:
        preview_img = np.rot90(preview_img, -rot_k)
        preview_img = cv2.resize(preview_img, (350, 450), interpolation=cv2.INTER_CUBIC)

    prev_c1, prev_c2 = st.columns([1, 1.4])
    with prev_c1:
        st.image(preview_img, channels="BGR", caption=f"최종 제출 미리보기 ({s_id}_{s_name}.jpg)", width=250)
        if st.button("🔃 사진 회전하기 (시계방향 90°)", use_container_width=True):
            st.session_state["rotation_deg"] = (st.session_state["rotation_deg"] + 90) % 360
            st.rerun()
            
    with prev_c2:
        st.markdown(
            f"**📌 선택된 제출 정보**\n"
            f"- **저장될 파일명:** `{s_id}_{s_name}.jpg`\n"
            f"- **선택한 스캔 형태:** {chosen_title}\n\n"
            f"미리보기 사진이 본인의 증명사진이 맞고 깔끔하게 스캔되었다면 아래 **최종 제출** 버튼을 눌러주세요.\n"
            f"마음에 들지 않으면 **다시 촬영 / 재업로드** 버튼을 눌러 새로 등록할 수 있습니다."
        )
        st.write("")
        if st.button("✅ 네, 지금 선택한 스캔본으로 최종 제출합니다!", type="primary", use_container_width=True):
            with st.spinner("선생님의 구글 드라이브로 증명사진을 안전하게 전송 중입니다..."):
                try:
                    is_success, buffer = cv2.imencode(".jpg", preview_img)
                    if is_success:
                        f_name = f"{s_id}_{s_name}.jpg"
                        upload_via_gas(buffer.tobytes(), f_name, target_gas_url)
                        st.session_state["final_uploaded"] = (f_name, preview_img)
                        st.session_state["photo_groups"] = None
                        st.rerun()
                    else:
                        st.error("이미지 인코딩 중 오류가 발생했습니다.")
                except Exception as e:
                    st.error(f"구글 드라이브 전송 중 오류가 발생했습니다: {e}")
                    
        if st.button("🔄 아니오, 사진을 다시 촬영 / 재업로드할게요", type="secondary", use_container_width=True):
            reset_scan_state()
            st.rerun()

# --- 최종 업로드 완료 화면 ---
if st.session_state["final_uploaded"] is not None:
    st.markdown("---")
    f_name, final_img = st.session_state["final_uploaded"]
    st.success(f"🎉 **[{f_name}]** 증명사진이 선생님 구글 드라이브에 성공적으로 제출되었습니다! 수고하셨습니다.")
    st.image(final_img, channels="BGR", caption=f"최종 제출 완료된 사진 ({f_name})", width=250)
    if st.button("다른 사진 다시 제출하기"):
        reset_scan_state()
        st.rerun()

# 학생 화면 하단 관리자 복귀 링크
if current_code and not st.session_state.get("is_teacher_creator", False):
    st.markdown("<br><hr><center><small style='color: gray;'>선생님이신가요? <a href='.' style='color: gray; text-decoration: underline;'>새 링크 만들기</a></small></center>", unsafe_allow_html=True)
