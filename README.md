# foam-cell-analysis

気泡インスタンスセグメンテーションアプリ（データ準備・モデル学習・モデル比較/リリース・本番推論）。
文書の入口は [docs/index.html](docs/index.html) です。仕様書は [docs/specs/](docs/specs/) を参照。
利用者向けの操作手順は [利用者マニュアル](docs/manual/index.html) を参照してください。

## セットアップ（pip のみ、Python 3.12）

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 学習・推論用の依存（torch / cellpose / optuna）

torch は CPU 版と CUDA 版で取得元が異なるため、**先に torch を入れてから** `[ml]` を入れる。

GPU なし（CPU 版）:

```
pip install -c constraints-ml.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints-ml.txt -e ".[ml]"
```

CUDA あり（例: CUDA 12.6。`nvidia-smi` のドライバ対応に合わせて cu126 などを選ぶ）:

```
pip install -c constraints-ml.txt torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -c constraints-ml.txt -e ".[ml]"
```

確認: `python -c "import torch; print(torch.cuda.is_available())"`

## 実行・テスト

```
foam-cell-analysis
pytest
```

既定では hybrid バックエンドを使い、`workspace/` に学習データと実験状態を保存します。
バックエンドは起動引数 `--backend mock|hybrid`、環境変数 `FOAM_BACKEND=mock|hybrid` の順に指定できます。
起動引数が環境変数より優先されます。どちらも指定しない場合は hybrid です。
画面確認だけを行う場合は `foam-cell-analysis --backend mock` を使います。

hybridでは学習と比較・評価の保存処理を、それぞれ永続化サービスが担当します。
学習用・検証用データ版の一覧・画像・マスクはDatasetStoreから実データを読み込みます。
取り込み・版登録などデータ準備の編集操作と本番推論はMockBackendの模擬処理です。
本番推論の実処理への接続は未完了です。

## 学習デバッグ用のダミーデータ

次のコマンドで `workspace/datasets/val_v000/` と `workspace/datasets/train_v000/` に、
512×512のグレースケール画像12枚と対応する16bitインスタンスマスク（背景0、気泡1〜25）を生成する。検証用版と学習用版を各1つ作り、学習用版は全画像を学習用にする。
分類A・Bが各6枚、各分類3グループ（各2枚）。各分類に品質「不良」1枚と、
気泡0個・品質「良」1枚を含む。不良画像はコントラストを下げ、ノイズを増やす。
気泡0個の画像は背景ノイズのみで、正解マスクは全画素0。
画像・マスクのパスとハッシュは `manifest.csv`、用途・分類・品質・生成シードは
`metadata.csv`、版情報は `dataset_info.json` に保存する。

```powershell
.venv\Scripts\python.exe scripts\generate_dummy_dataset.py --purpose val
.venv\Scripts\python.exe scripts\generate_dummy_dataset.py --purpose train --base-validation-version val_v000
```

検証用版を先に生成し、その版を基準にした学習用版を続けて生成する。既存フォルダは上書きしない。別の場所へ再生成する場合は `--output <保存先>` を指定する。
既定シードは学習用版が42、検証用版が1042（`--seed` で変更可能）。`group_id` と取り込み元フォルダで同じグループを示す。
分類別・グループ単位の交差検証は2〜3分割で確認する（各分類3グループのため）。
生成物はGit管理外。データ準備画面での取り込み・版登録は MockBackend の模擬処理で、確定済みデータの読み込みは DatasetStore に接続済み。本番推論は MockBackend の模擬処理で、実モデルへの接続は未完了。
