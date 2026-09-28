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
