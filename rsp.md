```python
# 숫자 손글씨의 경우 이미지 크기가 28x28 이었기 때문에, 우리의 가위, 바위, 보 이미지도 28x28로 만들어야 합니다.
# 이를 위해서  PIL 라이브러리 사용

from PIL import Image
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
```


```python
# 이미지 크기 조절 함수
def resize_images(img_path):
    images=glob.glob(img_path + "/*.jpg")  
    print(len(images), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in images:
        old_img=Image.open(img) # image.open : 이미지 불러오기
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG") # save 꼭 해야한다!
        # ANTIALIAS : 높은 해상도의 사진 또는 영상을 낮은 해상도로 변환하거나 나타낼때
        #깨진 패턴의 형태로 나타나게 되는데
        #이를 최소화 시켜주는 방법을 안티엘리어싱
    print(len(images), " images resized.")
```


```python
# 가위 이미지 resize
# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들이자
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
resize_images(image_dir_path)

print("scissor image resize complete!")
```

    1427  images to be resized.
    1427  images resized.
    scissor image resize complete!



```python
# 바위 이미지 resize
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"
resize_images(image_dir_path)

print("rock image resize complete!")
```

    1432  images to be resized.
    1432  images resized.
    rock image resize complete!



```python
# 보 이미지 resize
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"
resize_images(image_dir_path)

print("paper image resize complete!")
```

    1427  images to be resized.
    1427  images resized.
    paper image resize complete!



```python
# 데이터를 읽어올 수 있는 load_data() 함수 정의

def load_data(img_path, number_of_data):
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open
                       (file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("이미지 개수는", idx,"입니다.")
    return imgs, labels
```


```python
# train 데이터 불러오자

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path, number_of_data=4286) # 가위바위보 이미지 개수 총합에 주의하자
# 4286장!!
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
```

    이미지 개수는 4286 입니다.
    x_train shape: (4286, 28, 28, 3)
    y_train shape: (4286,)



```python
# 이미지 확인해보자

plt.imshow(x_train[4006])
print('라벨: ', y_train[4006])
```

    라벨:  2



    
![png](output_7_1.png)
    



```python
# 딥러닝 네트워크 설계

# Hint! model의 입력/출력부에 특히 유의해 주세요.
# 가위바위보 데이터셋은 MNIST 데이터셋과 어떤 점이 달라졌나요?
tf.random.set_seed(1234)  # seed()를 사용함으로써 동일한 결과를 얻자
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3))) 
# 16 : 얼마나 다양한 이미지의 특징을 볼 것인가?
# (28,28,3) : 입력이미지의 형태

model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
# 32 : 얼마나 다양한 이미지의 특징을 볼 것인가?

model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
# 32 : 분류기 알고리즘을 얼마나 복잡하게 할 것인가?

model.add(keras.layers.Dense(3, activation='softmax'))
# 3 : 최종 분류기의 class 수

print('Model에 추가된 Layer 개수: ', len(model.layers))
model.summary()
```

    Model에 추가된 Layer 개수:  7
    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_10 (Conv2D)           (None, 26, 26, 32)        896       
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 1600)              0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 64)                102464    
    _________________________________________________________________
    dense_11 (Dense)             (None, 3)                 195       
    =================================================================
    Total params: 122,051
    Trainable params: 122,051
    Non-trainable params: 0
    _________________________________________________________________



```python
#  x_train 학습 데이터로 딥러닝 네트워크를 학습

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train_norm, y_train, epochs=10)
```

    Epoch 1/10
    134/134 [==============================] - 1s 4ms/step - loss: 1.0642 - accuracy: 0.4067
    Epoch 2/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.5663 - accuracy: 0.7671
    Epoch 3/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.2910 - accuracy: 0.9022
    Epoch 4/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.2236 - accuracy: 0.9322
    Epoch 5/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.1201 - accuracy: 0.9655
    Epoch 6/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.0636 - accuracy: 0.9856
    Epoch 7/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.0528 - accuracy: 0.9881
    Epoch 8/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.0278 - accuracy: 0.9934
    Epoch 9/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.0187 - accuracy: 0.9975
    Epoch 10/10
    134/134 [==============================] - 0s 3ms/step - loss: 0.0083 - accuracy: 0.9997





    <tensorflow.python.keras.callbacks.History at 0x7f1b68604f90>




```python
# test 데이터 가위 리사이즈

image_dir_path_test = os.getenv("HOME") + "/aiffel/rock_scissor_paper_test/scissor"
resize_images(image_dir_path_test)

print("scissor image resize complete!")
```

    100  images to be resized.
    100  images resized.
    scissor image resize complete!



```python
# test 데이터 바위 리사이즈

image_dir_path_test = os.getenv("HOME") + "/aiffel/rock_scissor_paper_test/rock"
resize_images(image_dir_path_test)

print("rock image resize complete!")
```

    100  images to be resized.
    100  images resized.
    rock image resize complete!



```python
# test 데이터 보 리사이즈

image_dir_path_test = os.getenv("HOME") + "/aiffel/rock_scissor_paper_test/paper"
resize_images(image_dir_path_test)

print("paper image resize complete!")
```

    100  images to be resized.
    100  images resized.
    paper image resize complete!



```python
# test data 불러오기

image_dir_path_test = os.getenv("HOME") + "/aiffel/rock_scissor_paper_test"
(x_test, y_test)=load_data(image_dir_path_test, number_of_data=300) # 가위바위보 이미지 개수 총합에 주의하자
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    이미지 개수는 300 입니다.
    x_test shape: (300, 28, 28, 3)
    y_test shape: (300,)



```python
# 확인
# 데이터 1개 출력해서 확인
plt.imshow(x_test[85])
print('\n라벨: ', y_test[85])
```

    
    라벨:  0



    
![png](output_14_1.png)
    



```python
# 모델 테스트
test_loss, test_accuracy = model.evaluate(x_test_norm,y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

    10/10 - 0s - loss: 0.0813 - accuracy: 0.9667
    test_loss: 0.08132407814264297 
    test_accuracy: 0.9666666388511658



```python

```


```python

```


```python

```
