import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# 검색 할 샘플 사진을 로드 후 인코딩
kim_image = face_recognition.load_image_file("asset/images/kim.jpg")
kim_face_encoding = face_recognition.face_encodings(kim_image)[0]

# 검색 할 얼굴 인코딩 및 이름 배열 생성
known_face_encodings = [
    kim_face_encoding,
]
known_face_names = [
    "Kim Yunwoong",
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name != "Unknown":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # 얼굴에 Bounding Box를 그립니다
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # 얼굴 하단에 이름 레이블을 그립니다
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()