# python 機会学習プログラミング達人データサイエンティスト個人学習用

## CH2 分類問題
- 教師あり学習の線形分岐の基本概念。
- 勾配降下法のベクトル化実装
- 確率的勾配降下法によるオンライン学習に基づき、 ADALINEを効率よくトレーニング

## CH3 分類問題
> 機械学習のアルゴリズムのトレーニング５ステップ
- 1, 特徴量を選択する
- 2, 性能指標を選択する
- 3, 分類機と最適化アルゴリズムを選択する。
- 4, モデルの性能を評価する。
- 5, アルゴリズムを調整する.

> Sigmoid関数
- サンプルがクラス１に属する確率を表す.
- 尤度：結果から見られる条件のもっともらしさ.

> サポートベクターマシン(SVM)
- SVMの最適化の目的はマージンを最大かすること。マージンは、超平面（決定境界）とこの超平面にもっとも近いトレーニングサンプルとの間の距離として定義される。
- 超平面に最も近いトレーニングサンプルはサポートベクトルと呼ばれる.
> カーネルSVM
- 「カーネル」は２つのサンプル間の「類似性を表す関数」であると解釈できる.
- マイナス記号をつけているのは、距離の指標を反転させて類似度にするため。
> 決定木学習
- 得られた結果の意味を解釈しやすいかどうかに配慮する場合に魅力的なモデルである。
- 二分決定木で使用される不純度の指標または分割条件は、ジニ不純度・エントロピー・分類誤差の３つである.
> k近傍法：怠惰学習アルゴリズム
- k近傍法分類器(k-nearest neighbor classifier),略してKNNである.
> KNNのアルゴリズム
- １. kの値と距離指標を選択する。
- ２. 分類したいサンプルからk個のデータ店を見つけ出す。
- ３. 多数決によりクラスラベルを割り当てる

## CH4 データ前処理 - より良いトレーニングセットの構築
- CSV(Comma-Separated Values)

## CH5 次元削減でデータを圧縮する
> 内容
- 教師なしデータ圧縮での主成分分析（PCA)
- クラスの分離を最大化する教師あり次元削減法としての線形判別分析（LDA)
- カーネル主成分分析による非線形次元削減
> PCA(Principal Component Analysis)の目的
- 高次元データにおいて分散が最大となる方向を見つけ、元の次元と同じかそれよりも低い次元の新しい部分空間へ射影する。
> 線形判別分析(Linear Discriminant Analysis : LDA)
- LDAでは、データが正規分布に従っていることが前提となる。また、クラスの共分散行列が同じことと、特徴量が統計的にみて互いに独立していることも前提となる。

## CH6モデル評価とハイパーパラメータのチューニングのベストプラクティス
- Breast Cancer Wisconsin データセット、悪性腫瘍細胞と良性腫瘍細胞のデータセット
> ホールドアウト法
- 機械学習のモデルの汎化性能を評価するために使用する.

## CH７アンサンブル学習
- MajorityVoteClassifier
- バギング
- ブーストラップ
## CH8器会学習の適用１ー感情分析














