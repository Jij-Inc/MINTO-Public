# ログ機能ガイド

Mintoライブラリのログ機能を使用すると、実験とランの実行状況をリアルタイムで監視できます。このガイドでは、ログ出力の制御方法について詳しく説明します。

## 基本的な使用方法

### ログ機能の有効化

実験作成時に `verbose_logging=True` を指定することで、ログ機能を有効にできます：

```python
from minto import Experiment

# ログ機能を有効にした実験の作成
exp = Experiment(
    name="my_experiment",
    verbose_logging=True  # これによりログが自動出力される
)

run = exp.run()
with run:
    run.log_parameter("solver_type", "OpenJij")
    run.log_parameter("num_reads", 1000)
    # 実行中に自動的にコンソールにログが出力される
```

**出力例：**
```text
[2025-07-17 10:36:51] 🚀 Starting experiment 'my_experiment'
[2025-07-17 10:36:51]   ├─ 🏃 Created run #0
[2025-07-17 10:36:51]       ├─ 📝 Parameter: solver_type = OpenJij
[2025-07-17 10:36:51]       ├─ 📝 Parameter: num_reads = 1000
[2025-07-17 10:36:52]   ├─ ✅ Run #0 completed (0.2s)
```

### ログ機能の無効化

デフォルトでは `verbose_logging=False` のため、ログは出力されません：

```python
# ログなしの実験（デフォルト）
exp = Experiment(name="silent_experiment")
# または明示的に無効化
exp = Experiment(name="silent_experiment", verbose_logging=False)
```

## ログ設定のカスタマイズ

### LogConfigクラス

`LogConfig` クラスを使用して、ログの表示方法を詳細に制御できます：

```python
from minto import Experiment
from minto.logging_config import LogConfig, LogLevel, LogFormat

# カスタムログ設定
config = LogConfig(
    enabled=True,           # ログ機能の有効/無効
    level=LogLevel.INFO,    # ログレベル
    format=LogFormat.DETAILED,  # ログフォーマット
    show_timestamps=True,   # タイムスタンプ表示
    show_icons=True,        # アイコン表示
    show_colors=True,       # カラー表示
    show_details=True,      # 詳細情報表示
    max_value_length=100    # 値の最大表示長
)

exp = Experiment(
    name="custom_experiment",
    verbose_logging=True,
    log_config=config
)
```

### ログレベル

`LogLevel` 列挙型で出力するログのレベルを制御できます：

```python
from minto.logging_config import LogLevel

# DEBUG: すべてのログを出力（最も詳細）
config = LogConfig(level=LogLevel.DEBUG)

# INFO: 一般的な情報ログを出力（デフォルト）
config = LogConfig(level=LogLevel.INFO)

# WARNING: 警告以上のログのみ出力
config = LogConfig(level=LogLevel.WARNING)

# ERROR: エラーログのみ出力
config = LogConfig(level=LogLevel.ERROR)

# CRITICAL: 重要なエラーのみ出力（最小）
config = LogConfig(level=LogLevel.CRITICAL)
```

### ログフォーマット

`LogFormat` 列挙型で出力形式を選択できます：

```python
from minto.logging_config import LogFormat

# SIMPLE: シンプルな形式
config = LogConfig(format=LogFormat.SIMPLE)
# 出力例: "Starting experiment 'test'"

# DETAILED: 詳細情報付き（デフォルト）
config = LogConfig(format=LogFormat.DETAILED)  
# 出力例: "[INFO] Starting experiment 'test' with 2 runs"

# MINIMAL: 最小限の情報のみ
config = LogConfig(format=LogFormat.MINIMAL)
# 出力例: "test: started"

# COMPACT: コンパクトな形式
config = LogConfig(format=LogFormat.COMPACT)
# 出力例: "test | started"
```

## 表示オプション

### タイムスタンプ制御

```python
# タイムスタンプ表示
config = LogConfig(show_timestamps=True)
# 出力: [2025-07-17 10:36:51] 🚀 Starting experiment

# タイムスタンプ非表示
config = LogConfig(show_timestamps=False)
# 出力: 🚀 Starting experiment
```

### アイコン制御

```python
# アイコン表示（デフォルト）
config = LogConfig(show_icons=True)
# 出力: 🚀 Starting experiment

# アイコン非表示
config = LogConfig(show_icons=False)
# 出力: Starting experiment
```

### カラー制御

```python
# カラー表示（デフォルト）
config = LogConfig(show_colors=True)
# 出力: カラー付きテキスト

# カラー非表示（プレーンテキスト）
config = LogConfig(show_colors=False)
# 出力: プレーンテキスト
```

### 値の表示長制御

```python
# 長い値を制限
config = LogConfig(max_value_length=50)

# 長い値をそのまま表示
config = LogConfig(max_value_length=None)
```

## グローバル設定

### configure_logging関数

すべての新しいログインスタンスに適用されるグローバル設定を行えます：

```python
from minto.logger import configure_logging
from minto.logging_config import LogConfig, LogLevel

# グローバルログ設定
configure_logging(
    enabled=True,
    level=LogLevel.DEBUG,
    show_timestamps=True,
    show_colors=False  # CI環境などでカラーを無効化
)

# この後作成される実験はすべてこの設定を使用
exp1 = Experiment(name="exp1", verbose_logging=True)
exp2 = Experiment(name="exp2", verbose_logging=True)
```

### get_logger関数

グローバル設定されたロガーインスタンスを取得できます：

```python
from minto.logger import get_logger, configure_logging

# グローバル設定
configure_logging(enabled=True, level=LogLevel.INFO)

# グローバルロガーの取得
logger = get_logger()

# 直接ログ出力
logger.log_experiment_start("direct_experiment")
logger.log_parameter("test_param", 42)
```

## 実用的な設定例

### 開発環境での設定

```python
# 開発時は詳細なログを表示
dev_config = LogConfig(
    level=LogLevel.DEBUG,
    format=LogFormat.DETAILED,
    show_timestamps=True,
    show_icons=True,
    show_colors=True
)

exp = Experiment(
    name="development_experiment",
    verbose_logging=True,
    log_config=dev_config
)
```

### 本番環境での設定

```python
# 本番環境では必要最小限のログ
prod_config = LogConfig(
    level=LogLevel.WARNING,
    format=LogFormat.COMPACT,
    show_timestamps=True,
    show_icons=False,
    show_colors=False
)

exp = Experiment(
    name="production_experiment",
    verbose_logging=True,
    log_config=prod_config
)
```

### CI/CD環境での設定

```python
# CI環境ではカラーなし、アイコンなし
ci_config = LogConfig(
    level=LogLevel.INFO,
    format=LogFormat.SIMPLE,
    show_timestamps=True,
    show_icons=False,
    show_colors=False,
    max_value_length=100
)

exp = Experiment(
    name="ci_experiment",
    verbose_logging=True,
    log_config=ci_config
)
```

### デバッグ用設定

```python
# 問題調査時は最大限の詳細ログ
debug_config = LogConfig(
    level=LogLevel.DEBUG,
    format=LogFormat.DETAILED,
    show_timestamps=True,
    show_icons=True,
    show_colors=True,
    show_details=True,
    max_value_length=None  # 値を切り詰めない
)

exp = Experiment(
    name="debug_experiment",
    verbose_logging=True,
    log_config=debug_config
)
```

## ソルバー実行時のログ

### 自動パラメータログ

`log_solver` メソッドを使用すると、ソルバーのパラメータと実行時間が自動的にログされます：

```python
def my_solver(param1, param2, secret_key):
    # ソルバーの実装
    return {"energy": -100, "samples": 1000}

run = experiment.run()
with run:
    # ソルバーのラップと実行
    wrapped_solver = run.log_solver(
        "my_solver", 
        my_solver,
        exclude_params=["secret_key"]  # 機密パラメータを除外
    )
    
    result = wrapped_solver(param1=10, param2="test", secret_key="hidden")
    # param1とparam2は自動でログされるが、secret_keyは除外される
```

**出力例：**
```text
[2025-07-17 10:36:51]       ├─ 🔧 Solver: my_solver
[2025-07-17 10:36:51]       ├─ 📝 Parameter: param1 = 10
[2025-07-17 10:36:51]       ├─ 📝 Parameter: param2 = test
[2025-07-17 10:36:52]       ├─ ⚡ Solver execution completed (0.8s)
```

## トラブルシューティング

### ログが表示されない場合

1. `verbose_logging=True` が設定されているか確認
2. `LogConfig.enabled=True` が設定されているか確認
3. ログレベルが適切に設定されているか確認

```python
# デバッグ用の確認コード
exp = Experiment(name="test", verbose_logging=True)
print(f"Verbose logging: {exp.verbose_logging}")
print(f"Logger enabled: {exp._logger.config.enabled}")
print(f"Log level: {exp._logger.config.level}")
```

### 出力が多すぎる場合

```python
# ログレベルを上げて出力を制限
config = LogConfig(level=LogLevel.WARNING)

# または特定の情報のみ表示
config = LogConfig(
    format=LogFormat.MINIMAL,
    show_details=False
)
```

### パフォーマンスへの影響

ログ機能は最小限のオーバーヘッドで設計されています：
- `verbose_logging=False` 時: オーバーヘッドなし
- `verbose_logging=True` 時: 実験処理時間に対して1%未満の影響

## まとめ

Mintoのログ機能は柔軟で強力な制御オプションを提供します：

- **基本制御**: `verbose_logging` パラメータで簡単にオン/オフ
- **詳細制御**: `LogConfig` クラスで表示方法を細かく調整
- **グローバル制御**: `configure_logging` で一括設定
- **環境対応**: 開発・本番・CI環境それぞれに最適化可能

適切な設定により、開発効率の向上とシステムの可視性確保を両立できます。