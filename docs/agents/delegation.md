# Codex への委任手順

[AGENTS.md](../../AGENTS.md) の「5. AI エージェントの役割分担」を補足する。

## 基本の流れ

1. Claude が設計を書く（`docs/design` または作業用のメモ）。
2. Luna に実装を委任する。
3. Claude が差分をレビューする。規模が大きい場合は Opus 5.5 のサブエージェントに任せる。
4. 指摘があれば Luna に修正させる（`codex exec resume` で文脈を引き継ぐ）。
5. 非 slow テストと ruff が通ったらコミットする。

修正と再レビューのやり取りは最大2往復とする。

## コマンド

```sh
# 実装（プロンプトは stdin で渡し、最終報告は -o で受け取る）
codex exec -m gpt-6-luna -c model_reasoning_effort='"medium"' -o report.md - < prompt.md

# 同じ文脈で続ける（session id はログの先頭付近に出る）
codex exec resume <session-id> -m gpt-6-luna -c model_reasoning_effort='"low"' - < fix.md

# 設計相談（read-only）
codex exec -m gpt-6-astra -s read-only -o astra.md - < question.md
```

### 注意点

- `codex exec resume` は元の実行時のモデルと sandbox を引き継がず、`~/.codex/config.toml` の設定に戻る。**毎回 `-m` を指定する。** read-only で動かしたい役割では、`-c sandbox_mode='"read-only"'` も付ける（resume には `-s` がない）。
- bash のタスクを止めても codex.exe のプロセスツリーは残り、スレッドを握り続ける（"already has an active writer"）。`taskkill /PID <node codex.js の pid> /T /F` で終了させる。
- Luna を並列で動かすときは、担当ディレクトリが重ならないようにする。作業場所は worker ごとに `git worktree` で分け、`PYTHONPATH=src` を付けて実行する（共有 .venv の editable install が worktree のコードを隠してしまうため）。
- reasoning effort は難しさに合わせる。単純な修正は low、通常の実装は medium、high は理由があるときだけ使う。

## Luna へのプロンプトの型

```markdown
## 目的
<何を実現するか。1〜3行>

## 対象
- 変更してよいファイル・ディレクトリ: ...
- 読むべき設計・仕様: docs/...

## 完了条件
- <動作として満たすこと>
- 変更箇所のテストファイルと ruff check / ruff format が通ること
- slow/ml テストは実行しない

## 禁止事項
- 対象外のファイルの変更、依頼範囲外のリファクタリング
- テストを弱める変更
- UI に内部値（ID、enum 名、生のパス、例外メッセージ）をそのまま出すこと
- 1行に複数の文を詰め込むこと

## 報告形式
変更点 / 実行したテスト / 未解決事項 / 設計から外れた点とその理由
```

Luna の初稿は、中身が薄い、1行に詰め込む、UI に内部値を出す、という傾向がある。プロンプトで先に釘を刺し、レビューでも重点的に確認する。

## レビューの依頼方法（Claude のサブエージェント向け）

- 差分の範囲（`git diff <base>..HEAD`）と対象ファイルを明示し、「全体を見て」とは頼まない。
- GUI の変更は、QTest でユーザーの操作経路をたどる再現スクリプトと、スクリーンショットで確認させる。
- 再レビューでは、前回の指摘が直ったかの確認に限る。
- スクリーンショットは Windows ネイティブで描画する（`WA_DontShowOnScreen` を設定して `grab()`）。offscreen はフォントが置き換わるので、見た目のレビューには使わない（pytest には使ってよい）。アプリのテーマ（QSS）が適用されていることも確認する。
