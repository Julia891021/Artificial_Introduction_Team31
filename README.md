# Artificial_Introduction_Team31  
ppt連結: https://www.canva.com/design/DAFC-mGvXH4/y8mfPOsTQE_jhjtJewWeAA/view?utm_content=DAFC-mGvXH4&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink  

## Overview of the task
**Part 1 根據圖片辨別臉型**  
  
利用EfficientNet-b5模型，辨別圖片中的人是Heart, Oblong, Oval, Round, Square何種臉型  
  
**Part2 比對資料集中相同臉型的人的面部特徵 (計算歐基里德距離)，並推薦髮型**  

利用開源函式庫face_recognition API，計算目前預測圖片與其預測類別資料集中所有照片的五官特徵相似度，選擇五官特徵最為相似的三個人的髮型作為推薦髮型  
  
**Part3 將推薦髮型使用GAN模型套用至原圖上**  
  
利用GAN模型將原圖與推薦髮型套用  
參考論文: <https://arxiv.org/abs/2106.01505>  

## Prerequisite  
coding environment: Goolgle colab  
requirement.txt  
```dlib==19.18.0+zzzcolab20220513001918
gdown==4.4.0
ipython==8.4.0
numpy==1.21.6
opencv_python==4.1.2.30
Pillow==9.1.1
requests==2.23.0
scikit_image==0.18.3
scikit_learn==1.1.1
scipy==1.4.1
skimage==0.0
torch==1.11.0+cu113
torchvision==0.12.0+cu113
tqdm==4.64.0

```

## Usage 
**執行main.ipynb**  
Part1+Part2:原圖以及推薦髮型會先儲存到Barbershop-main中的unprocessed資料夾(分別命名為original_pic_[index], hair0_[index], hair1_[index], hair2_[index]，index代表其在test data中所屬的index值)，接著執行align_face.py(有包含在main.ipynb中)會將圖片儲存至Barbershop-main中的input資料夾，最後執行Barbershop-main中的main.py(有包含在main.ipynb中)，套用成果圖會儲存於Barbershop-main中的output資料夾

##  Hyperparamenters
**Split Train, Validation Set** -> val_size = 0.1  
**Efficientnet-B5**  
Efficientnet: 利用「複合式模型縮放」依照固定比例調整深度、寬度和解析度  
深度、寬度及解析度分別由常數 α, β, γ 及縮放係數 φ 來決定  
此模型利用multi-objective neural architecture search找出Baseline --  EfficientNet-B0  
將Baseline Model擴充一倍後找到適當的α, β, γ值，得到EfficientNet-B1，以此類推到EfficientNet-B7  
而我們的模型是使用EfficientNet-B5，參數量、深度、寬度、輸入解析度分別如下圖所示  
<img width="342" alt="image" src="https://user-images.githubusercontent.com/66251431/172796479-dcf33e08-eed9-4ead-8986-80de0dfa8771.png"></br>
模型訓練: batch_size = 10, n_epochs = 20, learning_rate = 0.001  




## Experiment Result
**Part 1 根據圖片辨別臉型**  

Testing data預測成效  
![image](https://user-images.githubusercontent.com/66251431/172758721-54272edb-8bef-4fce-b60b-dd6ec8ea08b9.png)<br/>
  
部分預測結果圖片呈現  
<img src="https://user-images.githubusercontent.com/66251431/172752578-c005ec2c-79cd-4712-8834-001e0ca07de2.png" width="400" height="400" alt="Testing data真實分類"/>
<img src="https://user-images.githubusercontent.com/66251431/172752615-53e098d3-8a20-404c-9f80-10752afc2efe.png" width="400" height="400" alt="Testing data預測分類"/><br/>

**Part2 比對資料集中相同臉型的人的面部特徵 (計算歐基里德距離)，並推薦髮型**  
原圖(分類為Oblong)  
<img src="https://user-images.githubusercontent.com/66251431/172763140-630b20af-bfaf-40a7-902b-0eefd8cf0500.png" width="200" height="200" alt="原圖"/><br/>
與資料集Hair Sample/Oblong中的圖片進行面部特徵比對(計算歐基裡德距離)，將最為相像的三個人的髮型作為推薦髮型  
※歐基里德距離愈小，表五官特徵相似度愈高</br>
<img src="https://user-images.githubusercontent.com/66251431/172764125-6901d1a6-e51e-4ab2-93fa-662ee5de56ce.png" width="200" height="200" alt="推薦髮型1"/>
<img src="https://user-images.githubusercontent.com/66251431/172764212-7e032404-375f-4468-ba54-7178ab541b82.png" width="200" height="200" alt="推薦髮型2"/>
<img src="https://user-images.githubusercontent.com/66251431/172764448-33690e4f-78f3-46eb-9770-4a73bc58fdec.png" width="200" height="200" alt="推薦髮型3"/></br>

**Part3 將推薦髮型使用GAN模型套用至原圖上**  
<img src="https://user-images.githubusercontent.com/66251431/172767810-d8d3c9ed-c29a-430c-945d-f0d6cd214f29.png" width="200" height="200" alt="套用髮型1"/>
<img src="https://user-images.githubusercontent.com/66251431/172767939-7ae19ff1-1929-4995-a768-a7e8c2783a24.png" width="200" height="200" alt="套用髮型2"/>
<img src="https://user-images.githubusercontent.com/66251431/172767995-510c30c3-3418-4cf9-a384-861af38c9fcc.png" width="200" height="200" alt="套用髮型3"/></br>
