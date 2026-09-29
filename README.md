# foam-cell-analysis

気泡インスタンスセグメンテーションアプリ（データ準備・モデル学習・モデル比較/リリース・本番推論）。
仕様書は [docs/specs/](docs/specs/) を参照。

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

## 学習デバッグ用のダミーデータ

次のコマンドで `workspace/datasets/train_v000/` に512×512のグレースケール画像12枚と、
対応する16bitインスタンスマスク（背景0、気泡1〜25）を生成する。全画像が学習用。
分類A・Bが各6枚、各分類3グループ（各2枚）。各分類に品質「不良」1枚と、
気泡0個・品質「良」1枚を含む。不良画像はコントラストを下げ、ノイズを増やす。
気泡0個の画像は背景ノイズのみで、正解マスクは全画素0。
画像・マスクのパスとハッシュは `manifest.csv`、用途・分類・品質・生成シードは
`metadata.csv`、版情報は `dataset_info.json` に保存する。

```powershell
.venv\Scripts\python.exe scripts\generate_dummy_dataset.py
```

既存フォルダは上書きしない。別の場所へ再生成する場合は `--output <保存先>` を指定する。
既定シードは42（`--seed` で変更可能）。`group_id` と取り込み元フォルダで同じグループを示す。
分類別・グループ単位の交差検証は2〜3分割で確認する（各分類3グループのため）。
生成物はGit管理外。アプリへの読み込み・版登録はバックエンド実装時に接続する。
