

<h2 align="center">
  dlib face recognition
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.10-blue.svg"/>
  <img src="https://img.shields.io/badge/dlib-v19.23.0-blue.svg"/>
  <img src="https://img.shields.io/badge/face_recognition-v1.3.0-blue.svg"/>
</div>

간혹 TV나 영화를 보다보면 누가 누군인지 모를 정도로 닮아 혼란에 빠뜨리게 하는 연애인이 있습니다.

<div align="center">
  <a href="https://yunwoong.tistory.com/84" target="_blank" title="dlib, Python을 이용하여 얼굴 인식하는방법" rel="nofollow">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGV5xS%2FbtrrhFCMfNF%2FCZrJ7DtXXew3yKlkao7cSK%2Fimg.png" width="500" title="dlib face recognition" alt="dlib face recognition">
    </img>
  </a>
</div>

<div align="center">
  <img src="" width="70%">
</div>

다른 연애인 도플갱어 블로그나 얼굴 인식 기술을 소개하는 자료를 보면 항상 등장하는 사람이 있는데, 바로 Will Ferrell(배우)과 Chad Smith(뮤지션) 입니다.

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F343yJ%2FbtrriVx4FlE%2FZjFWbrDtX7SJm8SKNEyLR0%2Fimg.jpg" width="70%">
</div>

실제로 두 사람은 닮은 꼴로 유명해서 The Tonight Show라는 토크쇼에 출연했었습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/kMFec/btrrh52fUDD/LO0ajhYsm2ZQvdzqXVAxO1/img.gif" width="70%">
</div>

[이전 글](https://yunwoong.tistory.com/83)에서 얼굴을 검출하는 방법을 소개했었는데, 이번에는 **얼굴 고유한 특징을 찾아 구별해내는 얼굴 인식 기술**을 구현하는 방법을 소개하도록 하겠습니다.

------

우선, dlib이 이미 설치가 되어 있어야 합니다. 만약 설치되어 있지 않다면 [dlib 설치가이드](https://yunwoong.tistory.com/80)를 참고하시여 설치를 진행하시기 바랍니다.

#### **1. Install**

```python
pip install face_recognition
```

#### **2. Import Packages**

```python
from imutils 
import face_utils 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
import imutils 
import dlib 
import cv2 
import face_recognition 
 
known_face_encodings = [] 
known_face_names = []
```

#### **3. Function**

Colab 또는 Jupyter Notebook에서 이미지를 확인하기 위한 Function입니다.

```python
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

얼굴을 찾아 인코딩 후 비교하고 출력하는 Function 입니다. Fucntion에 대해 간략히 설명드리자면 이미지에서 얼굴을 찾고 찾은 영역의 얼굴을 인코딩합니다. 이렇게 찾은 얼굴의 인코딩 값을 반복적으로 수행하면서 찾으려고 등록했던 known_face_encodings 리스트와 비교합니다.

```python
def name_labeling(input_image):
    image = input_image.copy()
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    face_names = []
 
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
 
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
 
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
 
        face_names.append(name)
        
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != "Unknown":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
 
        cv2.rectangle(image, (left, top), (right, bottom), color, 1)
        cv2.rectangle(image, (left, bottom - 10), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 3, bottom - 3), font, 0.2, (0, 0, 0), 1)
        
    plt_imshow("Output", image, figsize=(24, 15))
```

찾을 얼굴을 등록하는 Function입니다.

```python
def draw_label(input_image, coordinates, label):
    image = input_image.copy()
    (top, right, bottom, left) = coordinates
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)
    cv2.putText(image, label, (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    return image
    
def add_known_face(face_image_path, name):
    face_image = cv2.imread(face_image_path)
    face_location = face_recognition.face_locations(face_image)[0]
    face_encoding = face_recognition.face_encodings(face_image)[0]
    
    detected_face_image = draw_label(face_image, face_location, name)
    
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    
    plt_imshow(["Input Image", "Detected Face"], [face_image, detected_face_image])
```

#### **4. Add face**

이미지 경로와 이름을 입력하면 얼굴 인코딩 값을 리스트에 추가합니다.

```python
add_known_face("asset/images/Boris_Johnson.jpeg", "Boris Johnson") 
add_known_face("asset/images/Angela_Merkel.jpeg", "Angela Merkel")
```

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp87AV%2FbtrrcuvLCwD%2F0G6qWlDBTwrF792Kl9cYxK%2Fimg.png" width="70%">
</div>

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FKrUkV%2Fbtrrd6VgZiH%2F8bFV2jsV5WBMCcFbNL08z1%2Fimg.png" width="70%">
</div>

#### **5. Face Recognition**

```python
known_face_encodings = []
known_face_names = []
 
test_image_path = 'asset/images/2021_g7.jpg'
test_image = cv2.imread(test_image_path)
 
if test_image is None:
    print('The image does not exist in the path.')
else:
    print('image loading complete.')
```

```python
name_labeling(test_image)
```

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmTjsK%2FbtrrhTgFehM%2FiFlN7dsJkHoKtQTmnjebLK%2Fimg.png" width="70%">
</div>

정확하게 잘 찾았네요.

<div align="center">
  <img src="/asset/images/img.gif" width="50%">
</div>

------

2019년 MWC에서도 얼굴 인식과 관련 된 주제를 소개하는 부스가 참 많았습니다.

당시 재미났던 경험은 얼굴을 인식하여 문을 열어주는 솔루션을 가지고 나온 일본 업체가 있었습니다. 인식까지는 잘됐습니다. 그런데 혹시 촬영된 사진으로도 열리는지 물어봤더니 절대 안된다 했는데...열렸..습니다..ㅎ

일본 업체는 황당했지만 이미 많은 얼굴 인식 솔루션에는 face spoofing을 방지하는 방법과 알고리즘은 많이 연구되었고 적용하고 있습니다.

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbkY6y3%2FbtrrdmKOGDX%2FvZAfZ030vbj0uwtC3II4kK%2Fimg.jpg" width="70%">
</div>
