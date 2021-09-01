# Pythonで機械学習して知らないキノコを毒死せずに食べる
###### tags: `OSK`

[TOC]

## はじめに
Pythonで機械学習をする。機械学習といっても、深層学習(ディープラーニング)ではない。

ゼミの課題で、実データをとってきて何か分析をしないといけない。何か面白いデータないかな...。

浅はかな気持ちで「データセット 面白い」でGoogle検索をかけてたら、毒キノコのデータに巡り合えたので、これを分析することにした。

発表ではKaggle(機械学習コンペティションサイト)からデータを持ってきてる人もいた。ここにもタイタニックのデータとか、面白そうなデータがあるみたい。

---

## データセット
ゼミで発表をしたので、スライドを引用しながら記事を進める。

![](https://i.imgur.com/9QD5Ta4.png)

### データ概要
Mushroom Data Set(https://archive.ics.uci.edu/ml/datasets/mushroom)

データ数は8124で、属性の数は22。
各行にはキノコの物理的特徴と、毒があるか(edible/poisonous)のデータが入っている。内訳は
- 食べられる(edible): 52%
- 有毒(poisonous): 48%

属性には次のようなものがある。
- 傘の形状（鐘, 円錐, 平…）
- 匂い（アーモンド, 腐敗臭…）
- 茎のサーフェス（繊維質, うろこ状, なめらか…）
- 生息地（草, 牧草地, 木…）
- ひだの間隔（狭い、離れた…）

### 項目の詳細
属性のすべては以下(リンクから引用)。
```
Attribute Information:

1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
4. bruises?: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
```

よく訳がわからないものもある。たとえば、`ring` という単語がやたら出てくるのだが、それが何を指すのかよくわからなかった。茎(?)にリングみたいなものが一般にみられるのか...?

それにしても、キノコの物理的特徴って22個も観測できるものなのですね。作成者のキノコに対する表現力は当たり前だがスゴイ。匂いの項目をアーモンドとか、スパイシーとか言語化して分類しているのもポイント高いと思う。

---

## 方針
データセットで予測された使い方にのっとってそのまま使う。物理的特徴を特徴量とし、edible/poisonous(食べられる/有毒)を予測することにする。

予測には線形判別分析を使うことにする。線形判別分析は、下に概要をまとめておく。

ついでに、どんな特徴を持つキノコが有毒なのか気になるので、予測に大きく影響を与える因子が何なのかも交えて考察することを目指すことにした。

---

## データの前処理

データの前処理として、次のようなことをした。

![](https://i.imgur.com/8JrnSEf.png)

### ダミー変数の導入
まず、データの値がほとんどすべて質的変数だったので、ダミー変数を導入した。ダミー変数は質的変数を量的変数に変換するために使われる処理で、下に概要をまとめておく。

今回のデータはほとんどが質的変数だし、項目数も多い(これとか)。
```
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
```
この列に対してダミー変数導入しただけで列は10個くらい増える。

結局、ダミー変数を導入したら項目数は**22→94**に変化した。

ちなみに、ダミー変数にするのはとても簡単にできるようになっていて、Pythonでは1行でやってくれる。(`pandas` ライブラリの`get_dummies` 関数を使っている。)

```python=
X = pd.get_dummies(X, drop_first=True)
```

:::info
最初にデータの欠損などについて調べるべきだったみたい。データの欠損があるデータは珍しくないので、欠損は平均で補ったり列ごと削除したりするらしい。なんでもできるやり方があるわけじゃなくて、個々のデータに沿って分析を進めるのが大事なんだな～。
:::

### データの置換
データの中で一つだけ量的変数とみなせそうなものがあった。それがこれ。
```
18. ring-number: none=n,one=o,two=t
```
さっきも言ったが `ring` が何を指すのかわかってない。僕の中ではキノコの茎にある部分にリングがよくあり、その個数だと踏んでいる。

これは量的変数とみなせそうなので、文字列(Pythonの`str` 型)から整数(Pythonの`int` 型)に変換した。
```python=
data = data.replace({'ring-number': {'n': 0, 'o': 1, 't': 2}})
```

ちなみに、分析がうまくいかなくなると思って目的変数(推論の対象)としている列も
```python=
data = data.replace({'edible': {'e': 0, 'p': 1}})
```
 と置き換えたけど、たぶんこれはなくても良かった(目的変数は数値に変換する必要がない)。
 
--- 
 
## 結果
- 交差検証(k=12)
    判別的中率: 0.968
    
- Leave-one-out 交差検証
    判別的中率: 0.99951
    
交差検証とLeave-one-out 交差検証は下に概要をまとめておく。

Leave-one-out 交差検証で精度は99.9％を超えました。精度高い！

交差検証はkを(データ数8124の約数の中から)適当に選んだのですが、そんなに多くなくてよかったっぽい(データが割と多いので)。

これだけデータがあれば一般的にはLeave-one-outをする意味もあんまりないはず。今回は精度が結構上がっているのはなんでなの？？

---

## どんなキノコが毒キノコなのか

判別得点が0以上でpoisonous, 0未満でedibleと判別していることが分かったので、判別得点の重みのなかで係数が高いものはpoisonousと判断することに大きく貢献していることになる。

計数が高いものを抜き出したら次のようなものが出てきた。

![](https://i.imgur.com/ewtOCfd.png)

匂い(スライドでは臭いと記載しているが特に意味はない)とかは刺激臭とか、毒キノコっぽい特徴が見て取れる。色に関してもビビッドな色が多く上がってきていて、感覚に合致する。

次に、係数が低いものを抜き出してみる。

![](https://i.imgur.com/0Ene25m.png)

ランキングがひだの色に占拠された。食用キノコらしい特徴が多いかというと、そうでもなかった。

ひだの色はほとんど同じ位置にあり、しかも係数がすごく近くなっているので、判別に影響をあまり与えない のではないかという風に考えられる。(edibleに大きく貢献しているわけではなさそう)

このランキングは、`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` の `coef_` 関数で判別係数を表示し、それをソートしている。つまり単純に判別係数が高い順/低い順に並べただけなので、判別に決め手となる属性が正しく抜き出せているかというと微妙かもしれない。

主成分分析とかしたら、因子の特定につながるのかな～と思いつつも、列数が多いしほとんどが0と1の列なのでズレてるような気もした。いい方法があれば誰か教えてほしい。

---

## まとめ
今回はキノコのデータセットを使って分析を行った。精度は申し分ないかなという感じで、考察(影響を与える因子)についてはもう少しいろいろなやり方ができそうという感じになった。

---

## 用語について 
### 線形判別分析
説明変数($x_1, x_2, ..., x_n$) から 目的変数( $y$ ) を予測する手法。2つ(基本的には)のクラスを1本の直線で分離するように学習する。

判別分析では判別得点 $u$ を算出して判別予測を行う。
$u$ の算出は

$(特徴量1) × w_1 + (特徴量2) × w_2 + ... + (特徴量n) × w_n$

のように行われる。そのため、説明変数は数値に変換する必要がある。
また、判別の推測は $u$ の正負によって判断される。

### 交差検証
機械学習では学習データと評価データを分ける習慣がある。これは、汎化性能(未知のデータに対して正しい予測ができる能力)を重視しているためである。

学習に使ったデータで性能が上がっても汎化性能があるとはいえないので、精度の評価には学習に使用していない別のデータ(多くの場合、前もって取っておく)を用いる。

しかし、データ数が少ない場合など、データを分割することは避けたいことがある。学習に多くのデータを使用したほうが精度が高くなることが予想されるためである。

このような場合に、交差検証を使う。今回使った$k$-分割交差検証は次のような手順をとる。

1. 全データを $k$ 個に分割
2. そのうち1つをテストデータにし、取っておく。ほかのすべてのデータで学習する
3. とっておいたテストデータで精度を評価する
4. これを $k$ 回繰り返し、すべてのデータがテストデータとして使われるようにする
5. 精度の平均をとり、モデルの性能とする

これにより、なるべく多くのデータを学習に割くことができる。

### Leave-one-out 交差検証
交差検証のうち, $k$ =(データ数) に設定したもの。
評価データの数が1になるので、学習データの数が(データ数-1)だけ確保できる。

### ダミー変数
この時、特徴は質的変数と量的変数に大別でき、それぞれ対応が異なる。
質的変数は順番に意味がないもの(天気や性別など)や、順番はあるが間隔に意味がないもの(服のサイズなど)があげられ、これらは単純に
```
晴：0, 曇り: 1, 雨: 2 ...
```
のように数値化してしまうと判別の妨げになる。
ここで、ダミー変数導入では、対象とする変数ごとに一意な列を作り、あてはまる場合は1, 当てはまらない場合は0を割り当てる。

たとえば、次のようなデータがあるとする。

||drink|size|
|:-:|:-:|:-:|
|0|Cola|S|
|1|Coffee|M|
|2|Cola|L|
|3|Coffee|S|
|4|Coffee|L|

ここで、drink, sizeのどちらも質的変数とみなし、ダミー変数を導入するとデータは次のようになる。

||drink_Cola|drink_Coffee|size_S|size_M|size_L|
|:-:|:-:|:-:|:-:|:-:|:-:|
|0|1|0|1|0|0|
|1|0|1|0|1|0|
|2|1|0|0|0|1|
|3|0|1|1|0|0|
|4|0|0|0|0|1|

なお、S/M/Lは数値に変換してもよさそうだが、
```
S: 0, M: 1, L: 2
```
のように割り当てると、LはMに比べて判別得点の重みが2倍乗算されることになる。このようなことが好ましくないのでここでは質的変数としている。
ml換算ができるなら、LはMの何倍体積があるのか適切に表現できるので、質的変数への変換は必要ないだろう。