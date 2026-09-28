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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
```

CUDA あり（例: CUDA 12.6。`nvidia-smi` のドライバ対応に合わせて cu126 などを選ぶ）:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ml]"
```

確認: `python -c "import torch; print(torch.cuda.is_available())"`

## 実行・テスト

```
foam-cell-analysis
pytest
```

## 学習デバッグ用のダミーデータ

次のコマンドで `workspace/datasets/train_v000/` に512×512のグレースケール画像5枚と、
対応する16bitインスタンスマスク（背景0、気泡1〜25）を生成する。全画像が学習用。
画像・マスクのパスとハッシュは `manifest.csv`、用途・分類・品質・生成シードは
`metadata.csv`、版情報は `dataset_info.json` に保存する。

```powershell
.venv\Scripts\python.exe scripts\generate_dummy_dataset.py
```

既存フォルダは上書きしない。別の場所へ再生成する場合は `--output <保存先>` を指定する。
既定シードは42（`--seed` で変更可能）。取り込み元は画像ごとに別グループとして記録する。
生成物はGit管理外。アプリへの読み込み・版登録はバックエンド実装時に接続する。
