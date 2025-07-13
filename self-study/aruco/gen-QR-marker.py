import numpy as np
import os
import cv2
from cv2 import aruco

# マーカーの保存先
dir_mark = "./marker"

# 生成するマーカーの種類
# サイズ：4x4~7x7,  枚数：50,100,250,1000
marker_type=aruco.DICT_4X4_50

# 生成するマーカー用のパラメータ
marker_num = 20 #個数
marker_size = 1000 #マーカーのサイズ

# 余白[%]
margin_right = 0.1
margin_left  = 0.1
margin_top   = 0.1
margin_bottom= 0.1

# id のプリント
font_size=1

# ディレクトリ作成
os.makedirs(dir_mark,exist_ok=True)

# マーカー種類を呼び出し
dict_aruco = aruco.getPredefinedDictionary(marker_type)

for id in range(marker_num) :
    # マーカー用背景画像（白）
    img_mark=np.ones((int(marker_size*(1+margin_top+margin_bottom)),
                      int(marker_size*(1+margin_left+margin_right))))*255
    
    # マーカーを書き込む場所の計算
    x1=int(margin_left*marker_size)
    x2=x1+marker_size
    y1=int(margin_top*marker_size)
    y2=y1+marker_size
    
    # マーカーの書き込み
    img_mark[x1:x2,y1:y2] = aruco.generateImageMarker(dict_aruco, id, marker_size)

    # ファイル名
    marker_name="aruco."+[i for i in aruco.__dict__.keys() if aruco.__dict__[i]==marker_type if "DICT" in i][0]+" - "+str(id).zfill(3)
    
    # ファイル名の画像への書き込み
    if font_size!=0:
        cv2.putText(img_mark,marker_name,(0,img_mark.shape[1]-3),cv2.FONT_HERSHEY_PLAIN,font_size*marker_size/1000,0,int(font_size*marker_size/1500))

    # 画像の保存
    path_mark = os.path.join(dir_mark, marker_name+".jpg")
    cv2.imwrite(path_mark, img_mark)
