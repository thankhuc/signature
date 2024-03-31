# signature
Công cụ tìm kiếm chữ ký bằng python
def search_image(query_image_path, target_image_path):
    # Đọc ảnh truy vấn và ảnh mục tiêu
    query_image = cv2.imread(query_image_path)
    target_image = cv2.imread(target_image_path)

    # Chuyển đổi ảnh thành grayscale để tìm kiếm
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Tạo bộ phân cảnh (feature extractor) và matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Tìm các keypoint và descriptor trong ảnh truy vấn và ảnh mục tiêu
    query_keypoints, query_descriptors = orb.detectAndCompute(query_gray, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target_gray, None)

    # So khớp các descriptor bằng matcher
    matches = bf.match(query_descriptors, target_descriptors)

    # Sắp xếp các kết quả theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)

    # Vẽ các kết quả khớp lên ảnh mục tiêu
    result_image = cv2.drawMatches(query_image, query_keypoints, target_image, target_keypoints, matches[:10], None, flags=2)

    # Hiển thị ảnh kết quả
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Gọi hàm tìm kiếm ảnh
search_image("query_image.jpg", "target_image.jpg")
