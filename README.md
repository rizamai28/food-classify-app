# このアプリについて
### アプリケーション名
「Food Health Checker」

### 概要
食べ物の画像を認識して健康度を判定するアプリです。  
あなたが撮った写真や画像をアップロードすることで、AIがどれだけ健康的な画像であるかを判定します。  
レスポンシブに対応してるのでスマホからもご覧いただけます。   
判定結果をぜひTwitterでツイートしてみましょう。

### URL
[food-classify-app.herokuapp.com/](https://food-classify-app.herokuapp.com/)  
※ 読み込みに時間がかかる場合があります。

### 利用方法
1. 画像をアップロードして判定ボタンを押します。(jpg, png, gif形式のいずれか)
2. 判定結果が５段階で評価されます。
3. 評価が高くなるような画像をアップロードしてみましょう！

### 作成した目的
コロナ禍で外出や運動をする機会が減り、以前より食事に気をつけないとすぐに太ってしまうため、そんな生活を少しでも見直すきっかけになればと思いこのアプリを作成しました。

# 使用技術
- Python 3.6.10
- Flask 1.1.2
- gunicorn 20.0.4
- h5py 2.10.0
- jinja2 2.10.0
- keras 2.3.1
- numpy 1.18.1
- pillow 7.0.0
- tensorflow 2.0.0
- werkzeug 1.0.1
- Google Colaboratory

# Webアプリの構成図
![概要図 drawio (1)](https://user-images.githubusercontent.com/56781357/150352521-d07c2b07-9a5c-4ce9-80f7-d9fac3d612d3.png)

# 訓練用モデル(CNN)について
- CNNのモデル構造
```
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(dense_size))
model.add(Activation('softmax'))

model.summary()
```
- モデル構造は、以下のように構成します。
- - -
畳み込み層1  
畳み込み層2  
Maxプーリング層1  
畳み込み層3  
畳み込み層4  
Maxプーリング層2  
全結合層1  
全結合層2 
- - -  
途中でドロップアウトを挟むことにより、モデルの汎化性能が向上します。
  - 畳み込み層1: `Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:])`
    - 画像のサイズ 32 × 32
    - フィルタ数 32
    - フィルタサイズ 3 × 3
    - ゼロパディング  `padding='same'` 
  - 畳み込み層2: `Conv2D(32, (3, 3))`
    - フィルタ数 32
    - フィルタサイズ 3 × 3
  - Maxプーリング層1: `MaxPooling2D(pool_size=(2, 2))`
    - プールサイズ 2 × 2  
  - 畳み込み層3: `Conv2D(64, (3, 3), padding='same')`
    - フィルタ数 64
    - フィルタサイズ 3 × 3
    - ゼロパディング `padding='same'`
  - 畳み込み層4: `Conv2D(64, (3, 3))`
    - フィルタ数 64
    - フィルタサイズ 3 × 3
  - Maxプーリング層2: `MaxPooling2D(pool_size=(2, 2))`
    - プールサイズ 2 × 2
  - 全結合層1: `Dense(512)`
    - ニューロン数 512
  - 全結合層2: `Dense(dense_size)` (今回のモデルでは、健康な画像か脂っこい画像を分類するため)  
    - ニューロン数 2
## 入力画像について
健康的な食事の画像と脂っこい食事の画像をそれぞれ200枚用意し、1割(20枚ずつ)はテストデータとして用いました。  
入力画像のサイズは 32 × 32 にリサイズしました。

## 学習について
- 最適化アルゴリズム: `optimizers="Adam"`
- エポック数 80
- バッチサイズ 10
- 損失関数 交差エントロピー誤差: `loss="categorical_crossentropy"`
### データ拡張について
以下のようにデータ拡張をすることで、少ないデータ量でもモデルに沢山のデータを訓練させることができます。
```
generator = ImageDataGenerator(
           rotation_range=0.2,
           width_shift_range=0.2,
           height_shift_range=0.2,
           shear_range=10,
           zoom_range=0.2,
           horizontal_flip=True)
```
`rotation_range`: &emsp;画像を回転する角度を調整します。  
`width_shift_range`: &emsp;画像を水平方向にシフトできます。  
`height_shift_range`: &emsp;画像を垂直方向にシフトできます。  
`shear_range`: &emsp;引っ張るような変換を加えることができます。  
`zoom_range`: &emsp;拡大縮小範囲を指定できます。  
`horizonal_flip`: &emsp;水平方向に反転することができます。

## 学習済みのモデル
学習済みのモデルは`food_classify_model.h5`としてローカルに保存しました。

## 学習の推移を表示
損失関数のグラフ    
![スクリーンショット 2022-01-21 0 28 34](https://user-images.githubusercontent.com/56781357/150369209-611b3402-459b-4de7-aa99-d19c655817b6.png)  
<br>
<br>
精度のグラフ  
![スクリーンショット 2022-01-21 0 30 21](https://user-images.githubusercontent.com/56781357/150369540-2eb0f07d-1b0b-4515-b037-555f9173e0e5.png)
<br>
<br>
上記のグラフよりエポック数80では損失関数が収束しきれてないことが分かります。
<br>
## 予測
以下は、テスト画像(健康的な食事の画像が２０枚, 脂っこい食事の画像が２０枚)の中からランダムに２０枚を予測した結果です。  
<br>
![名称未設定](https://user-images.githubusercontent.com/56781357/150371976-7a57b1f3-4e2a-4870-8e0d-81c88f1ca002.png)






