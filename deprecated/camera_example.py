import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense 파이프라인 구성
pipeline = rs.pipeline()
config = rs.config()

# 깊이 및 컬러 스트림 활성화
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

try:
    while True:
        # 파이프라인에서 프레임셋을 기다림
        frames = pipeline.wait_for_frames()

        # 프레임셋에서 깊이 프레임과 컬러 프레임을 추출
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 깊이 이미지를 컬러맵으로 변환
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 결과 이미지 표시
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', images)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 종료
    pipeline.stop()
