'''
Created on Aug 12, 2019

@author: Mr 7ven
'''
# import các thư viện cần thiết
import face_recognition
import argparse
import pickle
import os
import cv2
import random
from builtins import int
from numpy import number

# xây dựng hàm chứa các đối số cho chương trình
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,    help="đường dẫn đến file chưa tt mã hóa")
ap.add_argument("-i", "--image", required=True,    help="đường dẫn đến ảnh đầu vào")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="phương thức nhận diện,hog hoặc cnn")
args = vars(ap.parse_args())

# load dữ liệu đã được mã hóa và nhúng từ file pickle
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load ảnh đầu vào và chuyển ảnh từ BGR sang RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# tìm các điểm (x,y) để xây dựng hình chữ nhật sau khi detect tương ứng với mối khuôn mặt
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# khởi tạo danh sách tên cho mỗi khuôn mặt được phát hiện
names = []

# vòng lặp trên các mã đã được mã hóa
for encoding in encodings:
    # so sánh sự trùng khớp giữa khuôn mặt trên ảnh và khuôn mặt đã biết
    matches = face_recognition.compare_faces(data["encodings"],    encoding)
    name = "Unknown"

    # kiểm tra kết quả so sánh
    if True in matches:
        # tìm chỉ mục của tất cả các khuôn mặt phù hợp, sau đó khởi tạo một từ điển để đếm tổng số lần mỗi khuôn mặt được khớp
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        #  lặp qua các chỉ mục phù hợp và duy trì số đếm cho mỗi khuôn mặt được nhận dạng
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # xác định khuôn mặt được công nhận với số phiếu bầu lớn nhất
        name = max(counts, key=counts.get)
    
    # cập nhật lại danh sách tên
    names.append(name)

# lặp lại trên các khuôn mặt được nhận diện
for ((top, right, bottom, left), name) in zip(boxes, names):
    # viết tên khuôn mặt lên ảnh
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# hiển thị kết quả
cv2.imshow("Image", image)
# split đường dẫn thư mục lấy tên file ảnh
txt = args["image"];
data = txt.split("\\")
numb1 = len(data) - 1
print(data[numb1])
# thay đổi thư mục lưu ảnh
os.chdir("C:\\Users\\Mr 7ven\\eclipse-workspace\\face-recognition-opencv\\output")
# lưu ảnh
cv2.imwrite("Rs_" + data[numb1], image)
k = cv2.waitKey(0)
    
