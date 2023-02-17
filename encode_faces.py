'''
Created on Aug 12, 2019

@author: Mr 7ven
'''
# import các thư viện cần thiết
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# xây dựng hàm chứa các đối số cho chương trình
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="đường dẫn đến thư mục ảnh đầu vào")
ap.add_argument("-e", "--encodings", required=True, help="đường dẫn đến file ghi dữ liệu sau khi encoding")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="kiểu nhận diện: có thể là hog hoặc cnn")
args = vars(ap.parse_args())

# lấy đường dẫn đến hình ảnh đầu vào trong tập dữ liệu
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# khởi tạo danh sách để lưu kết quả sau khi mã hóa
knownEncodings = []
knownNames = []

# vòng lặp các hình ảnh trong đường dẫn
for (i, imagePath) in enumerate(imagePaths):
    # trích xuất tên người từ đường dẫn hình ảnh
    print("[INFO] processing image {}/{}".format(i + 1,    len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load ảnh đầu vào và chuyển nó từ ảnh BGR sang RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # tìm các điểm (x,y) để xây dựng hình chữ nhật sau khi detect tương ứng với mối khuôn mặt
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # mã hóa mỗi khuôn mặt
    encodings = face_recognition.face_encodings(rgb, boxes)

    # vòng lặp mã hóa
    for encoding in encodings:
        # thêm vào danh sách ban đầu các mã hóa và tên đã biết
        knownEncodings.append(encoding)
        knownNames.append(name)

# ghi dữ liệu sau khi mã hóa vào file encodings.pickle
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
