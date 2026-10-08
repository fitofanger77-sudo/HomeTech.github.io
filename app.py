import streamlit as st
import cv2
import numpy as np
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# 페이지 설정
st.set_page_config(page_title="증명사진 수집 시스템", page_icon="📸")

st.title("📸 증명사진 수집 및 자동 스캔")
st.write("학번과 이름을 입력하고 증명사진을 업로드하면, AI가 자동으로 수평을 맞추고 보정하여 선생님 클라우드에 안전하게 저장됩니다.")

# 학생 입력 폼
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("학번 (예: 10501)")
with col2:
    student_name = st.text_input("이름 (예: 홍길동)")

uploaded_file = st.file_uploader("증명사진 파일 업로드 (스마트폰 촬영 또는 파일 선택)", type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def process_single_photo(image):
    """학생이 올린 단일 사진을 스캔 보정하는 핵심 알고리즘"""
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 어두운 마우스패드 배경과 흰색 스캐너 배경 모두 대응하는 마스크 생성
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    _, mask_scan = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    _, mask_dark = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.bitwise_or(edged, cv2.bitwise_or(mask_scan, mask_dark))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_area = 0
    img_area = img_h * img_w
    
    for c in contours:
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        w, h = rect[1]
        if w == 0 or h == 0: continue
        if w > h: w, h = h, w
        area = w * h
        
        # 사진 영역 감지 조건
        if img_area * 0.05 < area < img_area * 0.98 and 0.6 < (w / h) < 0.95:
            if area > max_area:
                max_area = area
                box = cv2.boxPoints(rect)
                best_box = np.intp(box)
                
    if best_box is not None:
        ordered_pts = order_points(best_box)
        target_w, target_h = 350, 450
        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    else:
        # 외곽선이 모호할 경우 기본 규격 리사이즈
        warped = cv2.resize(image, (350, 450), interpolation=cv2.INTER_CUBIC)
        
    # 정수리 보호 및 테두리 마감 컷 (머리 위쪽은 2픽셀만 다듬어 정수리 보호)
    final_crop = warped[2:442, 6:344]
    final_image = cv2.resize(final_crop, (350, 450), interpolation=cv2.INTER_CUBIC)
    return final_image

def upload_to_google_drive(file_bytes, file_name):
    """선생님 구글 드라이브로 자동 업로드하는 함수"""
    drive_creds = dict(st.secrets["google_drive"])
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    creds = service_account.Credentials.from_service_account_info(drive_creds, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    
    folder_id = st.secrets["google_drive"]["folder_id"]
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype='image/jpeg', resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# 제출 버튼
if st.button("사진 제출 및 자동 스캔 완료", type="primary"):
    if not student_id or not student_name:
        st.warning("⚠️ 학번과 이름을 모두 입력해주세요!")
    elif not uploaded_file:
        st.warning("⚠️ 증명사진 파일을 업로드해주세요!")
    else:
        with st.spinner("사진을 분석하고 정밀 스캔하여 구글 드라이브에 안전하게 저장 중입니다..."):
            try:
                # 업로드된 이미지 읽기
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # 핵심 스캔 보정 실행
                processed_img = process_single_photo(image)
                
                # JPG 인코딩
                is_success, buffer = cv2.imencode(".jpg", processed_img)
                if not is_success:
                    st.error("이미지 변환 중 오류가 발생했습니다.")
                else:
                    # 파일명을 '학번_이름.jpg'로 자동 정리
                    file_name = f"{student_id}_{student_name}.jpg"
                    
                    # 구글 드라이브로 전송
                    upload_to_google_drive(buffer.tobytes(), file_name)
                    
                    st.success(f"🎉 [{student_id} {student_name}] 학생의 증명사진이 성공적으로 제출 및 자동 보정되었습니다!")
                    st.image(processed_img, channels="BGR", caption="선생님 드라이브에 저장된 최종 보정 사진")
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")
