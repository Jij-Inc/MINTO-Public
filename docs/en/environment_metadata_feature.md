# MINTO環境メタデータ自動収集機能

## 概要

MINTOライブラリに環境メタデータの自動収集機能を実装しました。この機能により、ベンチマーク実験の再現性が大幅に向上し、異なる環境での実験結果の比較が容易になります。

## 新機能

### 1. 環境メタデータの自動収集

実験作成時に `collect_environment=True`（デフォルト）を指定することで、以下の情報が自動的に収集されます：

#### 収集される情報
- **OS情報**: オペレーティングシステム名とバージョン
- **ハードウェア情報**: CPU情報、コア数、メモリ容量、アーキテクチャ
- **Python環境**: Pythonバージョン、実行パス、仮想環境
- **パッケージバージョン**: 最適化関連の主要ライブラリのバージョン
- **実行情報**: タイムスタンプ

### 2. 永続化サポート

環境メタデータは以下の形式で保存・読み込みが可能です：
- ディレクトリ形式での保存/読み込み
- OMMX アーカイブ形式での保存/読み込み

### 3. 便利なメソッド

#### `get_environment_info()`
実験の環境メタデータを辞書形式で取得

#### `print_environment_summary()`
環境情報の要約を見やすい形式で表示

## 使用例

### 基本的な使用方法

```python
import minto

# 環境メタデータ収集を有効にした実験（デフォルト）
experiment = minto.Experiment(
    name="my_benchmark",
    collect_environment=True
)

# 実験実行
with experiment:
    experiment.log_parameter("algorithm", "my_algorithm")
    experiment.log_parameter("result", 42.0)

# 環境情報の表示
experiment.print_environment_summary()

# 実験結果の表示
results = experiment.get_run_table()
print(results)

# 実験の保存（環境情報も自動的に含まれる）
experiment.save()
```

### 環境メタデータの無効化

```python
# 環境メタデータ収集を無効化
experiment = minto.Experiment(
    name="simple_experiment",
    collect_environment=False
)
```

### 保存された実験の読み込み

```python
# ディレクトリから読み込み
loaded_exp = minto.Experiment.load_from_dir("path/to/experiment")

# OMMX アーカイブから読み込み
loaded_exp = minto.Experiment.load_from_ommx_archive("experiment.ommx")

# 環境情報の確認
env_info = loaded_exp.get_environment_info()
if env_info:
    print(f"実験実行OS: {env_info['os_name']}")
    print(f"Pythonバージョン: {env_info['python_version']}")
```

## ベンチマーク実験での活用

環境メタデータ機能により、以下のような利点があります：

1. **再現性の確保**: 実験環境の詳細情報が自動記録される
2. **環境比較**: 異なるマシンでの実験結果を適切に比較可能
3. **デバッグ支援**: 環境差異による問題の特定が容易
4. **研究報告**: 論文やレポートに必要な環境情報を自動取得

## 実装詳細

### アーキテクチャ

- `minto/environment.py`: 環境情報収集のコア機能
- `minto/experiment.py`: Experimentクラスへの統合
- 自動収集は実験作成時に一度だけ実行（オーバーヘッド最小化）
- エラー耐性: 環境情報収集に失敗しても実験は継続

### 依存関係

- `psutil`: ハードウェア情報の詳細取得（オプション）
- 標準ライブラリのみでも基本機能は動作

### テスト

`tests/test_environment_metadata.py` に包括的なテストを実装：
- 環境メタデータ収集のテスト
- 無効化機能のテスト
- 永続化（保存/読み込み）のテスト
- OMMX アーカイブでの永続化テスト
- メソッド動作のテスト

## 今後の拡張可能性

1. GPU情報の収集
2. ネットワーク環境情報
3. Docker/コンテナ環境の検出
4. カスタム環境情報の追加
5. 環境差異の自動分析機能

## まとめ

この実装により、MINTOは最適化実験の再現性とトレーサビリティを大幅に向上させる環境メタデータ自動収集機能を提供します。研究者や開発者は、実験環境の詳細を手動で記録する必要がなくなり、より信頼性の高いベンチマーク実験を効率的に実施できるようになりました。
