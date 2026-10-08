import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
io_module = __import__('io')

# 페이지 설정
st.set_page_config(page_title="학생 증명사진 자동 정렬 스캐너", page_icon="📸")

st.title("📸 1학년 5반 증명사진 자동 스캐너")
st.write("스캔본 또는 사진을 업로드하면 AI가 자동으로 23장을 분리하고 수평을 맞춰줍니다.")

# 파일 업로드 컴포넌트
uploaded_files = st.file_uploader("사진 파일을 업로드하세요 (여러 장 가능)", type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'], accept_multiple_files=True)

if uploaded_files:
    if st.button("스캔 및 정렬 시작하기"):
        with st.spinner("이미지를 분석하고 정밀 스캔 중입니다... 잠시만 기다려주세요!"):
            
            # 임시 결과 저장용 메모리 ZIP 파일 생성
            zip_buffer = io_module.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                
                for uploaded_file in uploaded_files:
                    # 업로드된 파일을 OpenCV 형식으로 읽기
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                        
                    img_h, img_w = image.shape[:2]
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # --- 선생님의 핵심 스캔 알고리즘 적용 ---
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
                    edged = cv2.Canny(blur, 30, 150)
                    combined_mask = cv2.bitwise_or(thresh_adapt, edged)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    
                    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    photo_data = []
                    img_area = img_h * img_w

                    for c in contours:
                        hull = cv2.convexHull(c)
                        rect = cv2.minAreaRect(hull)
                        w, h = rect[1]
                        if w == 0 or h == 0: continue
                        if w > h: w, h = h, w
                        area = w * h
                        
                        if img_area * 0.005 < area < img_area * 0.95:
                            if 0.65 < (w / h) < 0.90:
                                cx, cy = rect[0]
                                box = np.intp(cv2.boxPoints(rect))
                                photo_data.append({'box': box, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                    # 중복 제거
                    photo_data.sort(key=lambda d: d['area'], reverse=True)
                    unique_photos = []
                    for d in photo_data:
                        is_duplicate = False
                        for u in unique_photos:
                            dist = np.hypot(d['cx'] - u['cx'], d['cy'] - u['cy'])
                            if dist < min(d['w'], u['w']) * 0.5:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            unique_photos.append(d)
                            
                    avg_h = sum(d['h'] for d in unique_photos) / max(len(unique_photos), 1)
                    unique_photos.sort(key=lambda d: (d['cy'] // max(avg_h * 0.5, 1), d['cx']))

                    for i, d in enumerate(unique_photos):
                        # 4점 정렬 함수
                        pts = d['box']
                        rect_pts = np.zeros((4, 2), dtype="float32")
                        s = pts.sum(axis=1)
                        rect_pts[0] = pts[np.argmin(s)] 
                        rect_pts[2] = pts[np.argmax(s)] 
                        diff = np.diff(pts, axis=1)
                        rect_pts[1] = pts[np.argmin(diff)] 
                        rect_pts[3] = pts[np.argmax(diff)]
                        
                        target_w, target_h = 350, 450
                        dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
                        
                        M = cv2.getPerspectiveTransform(rect_pts, dst)
                        warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)

                        # 마감 및 정수리 보호 컷
                        final_crop = warped[2:target_h-8, 6:target_w-6]
                        final_image = cv2.resize(final_crop, (350, 450), interpolation=cv2.INTER_CUBIC)

                        # 메모리 내 ZIP 파일에 이미지 추가
                        is_success, buffer = cv2.imencode(".jpg", final_image)
                        if is_success:
                            file_name = f"{os.path.splitext(uploaded_file.name)[0]}_{i+1:02d}.jpg"
                            zip_file.writestr(file_name, buffer.tobytes())

            st.success("✨ 스캔 및 정리가 완료되었습니다!")
            
            # 다운로드 버튼 제공
            st.download_button(
                label="📦 정돈된 증명사진 전체 다운로드 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="증명사진_최종완성본.zip",
                mime="application/zip"
            )
