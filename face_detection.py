import cv2
import mediapipe as mp
import os

# MediaPipe modules for drawing and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 이미지 파일 경로입니다. 실제 이미지 파일 경로로 변경하세요.
image_path = './data/S006\\L9\\E02\\17111302_S006_L09_E02_C5.jpg'

# FaceMesh 설정을 시작합니다.
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    
    # 이미지를 로드합니다.
    image = cv2.imread(image_path)
    scale_factor = 8

    # 새로운 이미지 크기를 계산합니다.
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_dimension = (new_width, new_height)

    # 이미지의 크기를 조정합니다.
    resized_image = cv2.resize(image, new_dimension, interpolation=cv2.INTER_LINEAR)
    
    # 이미지가 제대로 로드되었는지 확인합니다.
    if image is None:
        print(f"Error: The image at {image_path} cannot be read.")
        exit(1)

    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
    results = face_mesh.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # 랜드마크 감지가 성공했는지 확인합니다.
    if results.multi_face_landmarks:
        annotated_image = resized_image.copy()
        
        # 감지된 랜드마크를 그립니다.
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

            # 각 랜드마크의 좌표를 출력합니다.
            for idx, landmark in enumerate(face_landmarks.landmark):
                print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

        # 결과 이미지를 저장합니다. 
        output_image_path = os.path.join(os.getcwd(), 'jy_face_detect.png')
        cv2.imwrite(output_image_path, annotated_image)

        # 결과 이미지를 화면에 표시합니다.
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face landmarks detected.")
