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

2. `http://localhost:8888`にアクセスして、`notebooks`配下の `1.baseline.ipynb`を実行(`0.eda.ipynb`から実行してもよい。)

## 使用モデル, 特徴量

- MMBT(https://arxiv.org/pdf/1909.02950.pdf)

  - BERT ベースの画像とテキストを入力としたマルチモーダルモデル。以下の 2 特徴量を結合して、再度 BERT に入力している。

    - 画像特徴量抽出: ResNet152

    - テキスト特徴量: BERT

## 実施予定事項

- [ ] 学習した MMBT(もしくは今回のタスクで未学習のモデル)の中間出力取り出し、他の特徴量(ex. 文字数、人の数、動物の数など)と組み合わせて lightGBM などで学習(2.extract_feature.ipynb にて実施中)

- [ ] 画像と文字の類似度を特徴量に加える(https://qiita.com/sonoisa/items/d6db2f130fa9a4ce0c2c などを参考)
