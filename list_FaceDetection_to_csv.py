import cv2
import mediapipe as mp
import csv
import os
import face_to_list

# MediaPipe modules for drawing and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh

# Define the CSV file for landmarks.
csv_file_path = 'face_landmarks.csv'
image_files = face_to_list.get_image_files('./data/')
print(f"Total number of images: {len(image_files)}")

# Prepare the CSV file with the header.
with open(csv_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["image_name"] + [f"x_{i}" for i in range(468)] + [f"y_{i}" for i in range(468)] + [f"z_{i}" for i in range(468)])

# Initialize Face Mesh once before processing the images.
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
        
    # 각 이미지에 대한 랜드마크 감지 및 CSV 파일에 저장
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: The image at {image_file} cannot be read.")
            continue
        
        scale_factor = 4

    # 새로운 이미지 크기를 계산합니다.
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        new_dimension = (new_width, new_height)

        # 이미지의 크기를 조정합니다.
        resized_image = cv2.resize(image, new_dimension, interpolation=cv2.INTER_LINEAR)
        # 이미지가 제대로 로드되었는지 확인합니다.
        if resized_image is None:
            exit(1)
        
        # 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # 랜드마크 감지가 성공했는지 확인합니다.
        if results.multi_face_landmarks:
            annotated_image = resized_image.copy()
            print("detect landmark")
            
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

            # Display the image with landmarks.
            # cv2.imshow(f'Annotated Image - {os.path.basename(image_file)}', annotated_image)
            # cv2.waitKey(0)
        
            # 감지된 랜드마크를 CSV 파일에 저장
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                with open(csv_file_path, mode='a', newline='') as file:
                    csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([os.path.basename(image_file)] + landmarks)

    # Face Mesh 자원 해제
    print("랜드마크 추출 및 저장이 완료되었습니다.")
