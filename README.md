# foam-cell-analysis

気泡インスタンスセグメンテーションアプリ（データ準備・モデル学習・モデル比較/リリース・本番推論）。
仕様書は [docs/specs/](docs/specs/) を参照。

## セットアップ

```
uv sync                # 画面・データ管理の依存 + 開発ツール
uv sync --extra ml     # 学習・推論用（torch / cellpose / optuna）
uv run foam-cell-analysis
uv run pytest
```
