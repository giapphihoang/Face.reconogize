'''
Created on Aug 12, 2019

@author: Mr 7ven
'''
# import các thư viện cần thiết
import face_recognition
import argparse
import imutils
import pickle
import cv2

# xây dựng hàm chứa các đối số cho chương trình
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,    help="đường dẫn đến file pickle chứa thông tin mã hóa")
ap.add_argument("-i", "--input", required=True,    help="đường dẫn đến video đầu vào")
ap.add_argument("-o", "--output", type=str,    help="đường dẫn cho video đầu ra")
ap.add_argument("-y", "--display", type=int, default=1,    help="cho phép video trình chiếu hoặc không")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="phương thức nhận diện, hog hoặc cnn")
args = vars(ap.parse_args())

# load dữ liệu đã được mã hóa và nhúng từ file pickle
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# khởi tạo con trỏ tới video đầu vào và trình ghi video
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

# lặp qua các khung hình từ video
while True:
    # lấy khung hình tiếp theo
    (grabbed, frame) = stream.read()

    # nếu khung không được lấy, thì ta đã đến cuối luồng
    if not grabbed:
        break

    # chuyển đổi khung đầu vào từ BGR sang RGB, sau đó thay đổi kích thước của nó để có chiều rộng 750px (để tăng tốc độ xử lý)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # phát hiện các tọa độ (x, y) của các hộp giới hạn tương ứng với từng mặt trong khung đầu vào, sau đó tính toán các mã hóa khuôn mặt cho mỗi mặt
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # vòng lặp qua các khuôn mặt đã mã hóa
    for encoding in encodings:
        # so sánh sự trùng khớp giữa khuôn mặt trên ảnh và khuôn mặt đã biết
        matches = face_recognition.compare_faces(data["encodings"],    encoding)
        name = "Unknown"

        # kiểm tra kết quả so sánh
        if True in matches:
            # tìm chỉ mục của tất cả các khuôn mặt phù hợp, sau đó khởi tạo một từ điển để đếm tổng số lần mỗi khuôn mặt được khớp
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # lặp qua các chỉ mục phù hợp và duy trì số đếm cho mỗi khuôn mặt được nhận dạng
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # xác định khuôn mặt được công nhận với số phiếu bầu lớn nhất
            name = max(counts, key=counts.get)
        
        # cập nhật lại tên
        names.append(name)

    # vòng lặp các khuôn mặt đã nhận diện
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # thay đổi giá trị khung hộp
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # viết tên người đã nhân diện lên ảnh
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # viết tên lên video với VideoWriter
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,(frame.shape[1], frame.shape[0]), True)

    # viết tên lên khung hình trong video
    if writer is not None:
        writer.write(frame)

    # hiển thị video đầu ra
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

# đóng con trỏ đến video
stream.release()

# kiểm tra và đóng con trỏ sau khi viết lên video
if writer is not None:
    writer.release()