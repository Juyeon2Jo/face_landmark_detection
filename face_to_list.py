import os

def get_image_files(folder_path):
    """
    주어진 폴더 경로에서 모든 하위 폴더를 포함하여 이미지 파일의 경로를 찾아 리스트로 반환하는 함수입니다.

    Args:
    folder_path (str): 이미지 파일을 검색할 폴더의 경로입니다.

    Returns:
    list: 발견된 모든 이미지 파일의 전체 경로를 포함하는 리스트입니다.
    """
    image_files = []
    # 지원되는 이미지 파일 확장자
    image_extensions = ('.png', '.jpg', '.jpeg')

    # 주어진 폴더 경로에서 모든 파일을 나열하고 이미지 파일만 리스트에 추가
    for root, _, files in os.walk('./data/'):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files

# 함수 사용 예시
# 실제 경로로 변경해야 함
folder_path = './data/'
image_list = get_image_files(folder_path)
# print(image_list)
