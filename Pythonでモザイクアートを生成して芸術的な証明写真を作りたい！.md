# Pythonでモザイクアートを生成して芸術的な証明写真を作りたい！
###### tags: `OSK`

[TOC]

## ストーリー
### 課題との出会い
**「なんでもいいからプログラミングしてこい」** という課題が飛び込んできた。(ゼミで)

正確に言うと、

```
1. 4択クイズ のシステム作成
2. 売上予測のシミュレーション
3. 森林火災のモデル化
4. 感染症の対策効果の検証 …
```

みたいな感じでいくつか例が挙げられていて、具体的にこういう実装をしてくれとナビゲーションがある。僕は「指示のあるプログラムなんてぬるま湯だぜ！」と思ったので、**「10. 自由課題」** しか見えなかった。

それじゃ結局、最初の課題内容は嘘なのですが、とにかく自由課題に取り組むことにする。

### どうせなら、映えるもの
ゼミではプログラムを書いて、それをスライドにまとめて発表をしないといけない。それならきっと**ビジュアル**なものがいいだろう。

モザイクアートってプログラムで自動で作れそう、というアイデアが前々からあったので、いい機会だと思ってやってみることにした。

ちなみに、1~2日でできるっしょ！みたいな感じで高をくくってはじめたら結構苦戦した。

---

## 問題設定

問題設定はこんな感じ。

「好きな写真を入力として、モザイクアートを出力するプログラムを作成する。」

めちゃめちゃざっくりだけど、これが目標。

### 萎えたこと
作成に当たってモザイクアートについて調べてみたら普通にそういうWebアプリケーションあった。

作りたい画像と、素材にしたい写真をアップロードしたらモザイクアートを作ってくれる。**もう、これでいいじゃん...** と思って若干萎えたけど、それ以外のアイデアもなかったのでこのままやることにした。

:::success
ちなみに実装の記事も見つかった。
ゼミで、誰かが書いたプログラムを使うことは禁止(正確には、理解していなかったり説明できないコードを使うのはダメ)されてるので、それは参考程度にとどめた。
:::

### フェイクの出現
脱線ばかりしてるけど、もう1個。
作るにあたってモザイクアートをいろいろ解説してる記事と巡り合った。それがこちら。
:::success
[第1回 本来のフォトモザイクアートとは！？ | JDO-MosaicArt](https://www.photomosaicart-de-memorial.jp/successpoint_no1/)
:::


モザイクアートって「ターゲット画像を、素材を並べて再現するもの」ですよね。でも、中には**ターゲット画像を透かして浮かび上がらせている画像もある**らしい。それ、意味ないじゃん。素人でもわかる。

モザイクアート界にも外道ってあるんだな～としみじみした。

---

## 環境

:::info
別に大事じゃないので飛ばしてくれて構わないです
:::

以下の環境を使った。

- Windows 10 Home 2004 64bit
- Python 3.9.5
- Visual Studio Code 1.56.2

Pythonは画像処理で使われる有名なライブラリ「`OpenCV`」を動かすことができる。あとで機械学習用のデータセットを使うことになるけど、Python上で`torchvision`というライブラリを使って呼び出している。大量の画像データを使う今回のようなプロジェクトにPythonはぴったりだと思う。

---

## 今回使う画像
ゼミで発表をしたので、発表用スライドをところどころ引用する。

![](https://i.imgur.com/Rjr0ZR5.png)

**※ カッコの中身は画像サイズ**

### ターゲット画像 (画像左)
綺麗な風景の写真とかをモザイク化してもあまり面白くないので、証明写真を使うことにした。

発表では、自分の証明写真を出したら受けるかなと思ったけど微妙な空気になった。どうしてくれんの！

ということで、これが原画に相当する。これをモザイク化すると思ってほしい。

### 素材(画像右)

Pythonが得意とする機械学習ではビッグデータを扱う。素材にはたくさんの画像が必要だから、**機械学習用のデータセット**は適任だと思う。

`CIFAR-100`は $32 × 32$ の画像が6万枚も入ったデータセットで、データが多いことと画像サイズが小さいことからこれを使うことに決めた。

また、`CIFAR-100` は画像と正解ラベルとよばれるものがセットになっていて、正解ラベルは機械が学習するために使われるのだが、今回は必要ないので大量の画像部分だけを使わせていただいた。

---

## プログラムの流れ

少し具体的に、プログラムの流れを整理しておく。
![](https://i.imgur.com/5bQOGB4.png)

ターゲット画像は $768×1024$。それぞれのエリアを似た素材に置き換えたいので、まずはターゲット画像を分割する。CIFAR-100を素材としたとき、横に24分割、縦に32分割して、**計768枚の画像で敷き詰めることができる。**

それから、分割されたエリアと`CIFAR-100`を何かしらの方法で比較して、類似度を計算する。一番似ている素材で置き換えたいので、類似度が最も高かった素材は保存しておく(正確には、素材のIDを保存しておく)。

分割されたエリアのすべてに対して、一番似ている素材が見つかったら、あとは素材を並べてプロットする。

### 画像処理の概要

:::info
画像処理について、概要だけ記しておく。
必要ない人は読み飛ばしてほしい。
:::

画像は**ピクセル**を並べたものである。画像を拡大してみると、カクカクしているのがわかると思う。その最小単位、画素とも呼ばれるものがピクセル。

:::success
たとえば、素材に使った画像CIFAR-100では、縦32 × 横32のピクセルが集まることで画像が構成されている。

![](https://i.imgur.com/76oP2rw.png)
:::

そして、ピクセルは一つの色を表している。
一つの色はRGBの値などで表せるから、**画像は数字に変換できる** ということになる。

グレースケール画像では、1つのピクセルは0～255の256段階の値をとる。
一般的なカラーの画像では、**RGB値** で色を表現していて、R(赤)、G(緑)、 B(青) のそれぞれの値が256段階をとる。

:::success
たとえば、
赤は(R, G, B) = (255, 0, 0) ,
紫は(R, G, B) = (255, 0, 255)。
:::

1ピクセルがとる値は、$256^3$ で **1600万以上** にもなる。

画像処理をするときは、このように画像を数値に変換してどうこうすることが分かってもらえればいい。

### 画像の類似度、どう計算する？
作成の中で一番苦戦したのがこの部分で、なかなかカラー画像の比較手法が見つからなかった。

白黒ならまだしも、オレンジとピンクがどれくらい近いのかとか、水色とブラウンがどれくらい近いのかとか、判定するのってどうすればいいのかわからない。実際、白黒画像を比べる手法はいっぱい見つかったけどカラー画像に関してはなかなかヒットしない。

探し回った結果、[類似画像検索システムを作ろう](https://aidiary.hatenablog.com/entry/20091003/1254574041) こちらのサイトを参考に、**ヒストグラム**による比較手法を採用することにした。

### こんな感じで比べることにした
1. 各ピクセルは $256^3$ 通りで表されている。ここで、RGBそれぞれを$4$ 段階に減色する。これにより、1ピクセルが$4^3$ 通りで表されるようになる。以下、使用したコード。

```python=
# (0~255)の値を4段階(0~3)に減色
# 入力: int(0~255), 出力: int(0~3)
def decrease_color(value):
    if value < 64:
        return 0
    elif value < 128:
        return 1
    elif value < 196:
        return 2
    else:
        return 3
```

2. RGBのそれぞれが $4$ 段階、つまり $4^3$ の組み合わせで表されるピクセルを、式 $R×1 + G×4 + B×16$ で 整数 $𝑛 (0≤𝑛≤63)$ に変換する。ここまでで、1600万以上通りだった1ピクセルがざっくり64通りに分類できたことになる。

3. $𝑛 (0≤𝑛≤63)$ で表されるピクセルがいくつ入っているかをヒストグラム化する。
:::success
ヒストグラムは、0の値を取るピクセルがいくつあるか(最大1024個), 1の値を取るピクセルがいくつあるか(最大1024個), ・・・, 63の値を取るピクセルがいくつあるか(最大1024個) を表していて、値の総和は1024になる。(32×32の画像の総ピクセル数に対応する。)
:::

4. 画像(3×32×32)→ヒストグラム(1×64)に変換が完了した。
:::success
画像をこういうヒストグラムに変換できた。
横軸は $n$, 縦軸は 「画像内の、$n$ で表されるピクセルの数」$(0 ～ 1024)$ を表している。

![](https://i.imgur.com/FXuUFYn.png)
:::

5. `OpenCV` の `CompareHist()` というヒストグラム比較用の関数で、分割されたエリアと素材とを比較する。比較手法には、相関(`CV_COMP_CORREL`) を使用した。


(ほかにも使えそうな画像の比較手法があれば教えてほしい。)

---

## 生成結果

![](https://i.imgur.com/ohEG0eb.png)

こんな感じの画像が出力された。同じ画像使われまくってるので、重複なしでも作ったほうがいいかも。

![](https://i.imgur.com/Ng6SIWs.png)

ということで、重複を許すパターンと、許さないパターンで作った。自己評価としては、顔のパーツはさすがに厳しいけどスーツのところとかは結構いい感じかなあ。

今回はターゲット画像を768枚の素材を並べて再現しているけど、違和感ない画像を作るためにはもっと多くの画像を並べて作るのが大事かもしれない。

ちなみに実行時間はノートPCで、全部で17分くらいだった。

---

## 改良できそうな点
今回の手法では、画像を比較するとき**画像の位置の情報や形状の情報が失われている。** これは、「この画像にはこういう色のピクセルがいくつ入っている」という情報に圧縮してから比べているため。

たとえば、アイルランドの国旗とコートジボワールの国旗は同じヒストグラムになる。

![](https://i.imgur.com/VwR2jv2.png)
左: アイルランド, 右: コートジボワール

これがモザイクアートの生成に悪影響なのは言うまでもない。ということで、画像を特徴量に変換する方法には改良の余地が大いにある。

(なお、実際に改良するかは別の問題。)

---

## プログラム概要
### 予防線・反省
プログラムには至らない点が結構あると思う。特に、ターゲット画像は、いろいろな画像サイズに対応していない。ゼミでの発表後に「これって(証明写真以外の)ほかの画像もできるんですか？」みたいな質問された。「もちろんできます」って返したけど、本当は画像を整形する処理を追加しないと動きません。

それから、画像を数値に変換した後の形式を`torch.Tensor`型で統一したのですが、あまり賢くなかったかも... 別に機械学習系の関数を使うわけではなかったので、より広く使われている`numpy.adarray`型で進行したほうがスムーズだったかなと思った。

### 各ファイルについて
`utils.py` では複数回使う関数をまとめている。
`preprocessing.py` では素材画像のヒストグラム化を、(何度もすることを防ぐため)あらかじめやっておき、保存しておく。(動かすと `cifar-100`, `cifar100object.pickle` が保存される。)
`main.py`ではターゲット画像のヒストグラム化～類似度計算～画像のプロットみたいな処理をしている。



### フォルダ構成
.
├── `target_image` 
│ &emsp;&emsp;└─ `idphoto.jpg`
├── `utils.py`
├── `preprocessing.py`
└── `main.py`

### プログラム全文
`utils.py`
```python=
import torch
import numpy as np


def decrease_color(value):
    """
    (0~255)の値を4段階(0~3)に減色
    入力: int(0~255), 出力: int(0~3)
    """
    if value < 64:
        return 0
    elif value < 128:
        return 1
    elif value < 196:
        return 2
    else:
        return 3


def convert_image_to_histogram(image):
    """
    (3, 32, 32) の画像を (1, 32, 32) に変換, その後,ヒストグラム(64次元ベクトル) に変換
    値は64段階(0~63)
    R(0~3)*1 + G(0~3)*4 + B(0~3)*16で算出
    入力: torch.Tensor, 出力: torch.Tensor
    """
    converted_image = np.empty((32, 32), dtype=np.int)
    for i in range(32):
        for j in range(32):
            pixel_value = 0
            for k in range(3):
                pixel_RGB_value = decrease_color(image[k, i, j].item())
                pixel_value += pixel_RGB_value * (4**k)
            converted_image[i][j] = pixel_value
    histogram = np.empty(64, dtype=np.int)
    for i in range(64):
        num = np.count_nonzero(converted_image == i)
        histogram[i] = num
    histogram = torch.from_numpy(histogram)
    return histogram
```

`preprocessing.py`

```python=
# coding: utf-8
import torchvision
import torchvision.transforms as transforms
import pickle
import time
from tqdm import tqdm
from utils import *


def main():
    photo_parts = torchvision.datasets.CIFAR100('./cifar-100',train=True, download=True, transform=transforms.ToTensor())
    photo_parts_histograms = [] # list (each item = torch.Tensor)
    for i in tqdm(range(50000)):
        histogram = convert_image_to_histogram(photo_parts[i][0]* 255)
        photo_parts_histograms.append(histogram)
    with open('cifar100object.pickle',mode='wb') as f:
        pickle.dump(photo_parts_histograms, f)

if __name__ == '__main__':
    t1 = time.time()
    main() 
    t2 = time.time()
    elapsed_time = t2 - t1
    print(f'elapsed_time: {elapsed_time}')
```

`main.py`

```python=
# coding: utf-8
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import pickle
import time
from tqdm import tqdm
from utils import *


def preprocess_target_image():
    """
    target_image の読み込みと整形
    入力: なし, 出力: torch.Tensor, torch.Tensor, list(値: torch.Tensor)
    """
    with Image.open('target_data/idphoto.jpg') as img:
        target_image = np.expand_dims(np.asarray(img, np.float32), axis=0)
    target_image = torch.as_tensor(target_image) # target_image.size() = (1, 1024, 768, 3)
    target_image = target_image.unfold(1, 32, 32).unfold(2, 32, 32) # target_image.size() = (1, 32, 24, 3, 32, 32)
    target_image = target_image.reshape(-1, 3, 32, 32) # target_image.size() = (768, 3, 32, 32)

    photo_parts = torchvision.datasets.CIFAR100('./cifar-100',
        train=True, download=True, transform=transforms.ToTensor())
    with open('cifar100object.pickle', mode='rb') as f:
        photo_parts_histograms = pickle.load(f)
    return target_image, photo_parts, photo_parts_histograms


def search_similar_histogram(target_image, search_num, photo_parts_histograms, no_duplication_flag):
    """
    target_imageの全エリアに対してヒストグラムに変換 => search_num 回の間 photo_parts_histogram と比較
    入力: (torch.Tensor, int, list), 出力: int
    """
    most_similar_image_ids = []
    for i in tqdm(range(768)):
        most_similar_image_id = 0
        max_similarity = 0
        target_area = target_image[i]
        target_area_histogram = convert_image_to_histogram(target_area)
        target_area_histogram = target_area_histogram.numpy().astype(np.float32)
        for j in range(search_num):
            if no_duplication_flag and j in most_similar_image_ids:
                continue
            photo_parts_histogram = photo_parts_histograms[j].numpy().astype(np.float32)
            similarity = cv2.compareHist(target_area_histogram, photo_parts_histogram, cv2.HISTCMP_CORREL) # 相関
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image_id = j
        most_similar_image_ids.append(most_similar_image_id)
    return most_similar_image_ids

def plotimage(ids, data):
    """
    ids からモザイクアートを生成し、png形式で保存
    入力: (list, list), 出力: なし
    """
    generated_image = torch.Tensor(768, 3, 32, 32)
    for i, item in enumerate(ids):
        generated_image[i] = data[item][0]
    torchvision.utils.save_image(generated_image, "generated.png", nrow=24, padding=0) # (32, 24, 3, 32, 32) で保存


def main():
    search_num = 50000
    no_duplication_flag = False # True: フォトパーツの重複を許容しない, False: 許容する
    target_image, photo_parts, photo_parts_histograms = preprocess_target_image()
    most_similar_image_ids = search_similar_histogram(target_image, search_num, photo_parts_histograms, no_duplication_flag)
    plotimage(most_similar_image_ids, photo_parts)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    elapsed_time = t2 - t1
    print(f'elapsed_time: {elapsed_time}')
```
## 参考文献

- 類似画像検索システムを作ろう（C++とOpenCV）(https://aidiary.hatenablog.com/entry/20091003/1254574041)
- Python + OpenCVで画像の類似度を求める(https://qiita.com/best_not_best/items/c9497ffb5240622ede01)
- CIFAR-100：物体カラー写真（動植物や機器、乗り物など100種類）の画像データセット(https://www.atmarkit.co.jp/ait/articles/2006/15/news036.html)
- PyTorchで画像を小さいパッチに切り出す方法(https://blog.shikoan.com/pytorch-extract-patches/)








