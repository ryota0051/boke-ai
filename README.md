## 検証環境

- OS: ubuntu 22.04.1 LTS

- GPU: NVIDIA GeForce RTX 2060 SUPER

- CPU: Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

- メモリ: 32 GB

## 環境構築

0. `https://www.nishika.com/competitions/33`から以下のファイルを指定の場所に配置

   - train.csv => `./dataset/csv`直下

   - test.csv => `./dataset/csv`直下

   - train.zip => 展開して、`./dataset/imgs/train`直下に画像を配置(train.zip 配下に直接画像が置かれているので、それを全て`./dataset/imgs/train`に配置する。)

   - test.zip => 展開して、`./dataset/imgs/train`直下に画像を配置

   - sample_submission.csv => `./dataset/csv`直下

1. Docker コンテナ起動

   - gpu の場合

     ```
     docker-compose up
     ```

   - cpu の場合

     ```
     docker-compose -f docker-compose-cpu.yml up
     ```

2. `http://localhost:8888`にアクセスして、`notebooks`配下の notebook を以下の順番で実行

   1. `1.4 MMBT_stratified_kfold_model.ipynb`

   2. `2.1 extract_feature.ipynb`

   3. `2.3 train_extracted_feature_with_stacking.ipynb`

## 使用モデル, 特徴量

1. 下記の 3 つのモデルをスタッキングさせたモデル

- MMBT(https://arxiv.org/pdf/1909.02950.pdf)

  - BERT ベースの画像とテキストを入力としたマルチモーダルモデル。以下の 2 特徴量を結合して、再度 BERT に入力している。

    - 画像特徴量抽出: ResNet152

    - テキスト特徴量: BERT

- lightGBM

  特徴量は、下記を使用

  - 学習した MMBT の中間出力

  - 文章長

  - 文章内に人を意味する単語が入っているか(固有表現抽出を使用)

  - tfidf の各種統計量

  - coco データで学習済みの yolov5 で検出した物体数

- RandomForest

  特徴量は、lightGBM のものと同様のものを使用
