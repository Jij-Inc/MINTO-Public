# MINTOユーザーガイド：初心者からエキスパートまで

## はじめに

### インストールとセットアップ

```bash
pip install minto
```

MINTOは以下を含む依存関係を自動的に処理します：
- OMMX（Open Mathematical Modeling Exchange）
- JijModeling（数理モデリングフレームワーク）
- Pandas（データ分析）
- NumPy（数値計算）

### 最初の実験

```python
import minto

# 自動環境追跡を有効にして実験を作成
experiment = minto.Experiment(
    name="my_first_optimization",
    collect_environment=True  # デフォルト：システム情報をキャプチャ
)

# 実験レベルのパラメータをログ（すべてのランで共有）
experiment.log_global_parameter("algorithm", "greedy")
experiment.log_global_parameter("max_iterations", 1000)

# 最適化イテレーションを作成して実行
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_parameter("objective_value", 42.5)
    run.log_parameter("solve_time", 0.123)

# 結果を保存して表示
experiment.save()
print(experiment.get_run_table())
```

## API設計の哲学

MINTOは実験レベルとランレベルのデータを明示的に分離します：

- **実験レベルデータ**：すべてのランで共有（問題、インスタンス、グローバルパラメータ）
- **ランレベルデータ**：個々のランイテレーションに固有（ソリューション、ランパラメータ）

### 実験レベルメソッド
これらのメソッドはすべてのランで共有されるデータを保存します：

- `experiment.log_global_parameter()` - 実験全体のパラメータ用
- `experiment.log_global_problem()` - 問題定義用
- `experiment.log_global_instance()` - 問題インスタンス用
- `experiment.log_global_config()` - 設定オブジェクト用

### ランレベルメソッド
これらのメソッドは個々のランに固有のデータを保存します：

```python
# 明示的なrunオブジェクトを作成
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_solution("sol", solution)
    run.log_sampleset("samples", sampleset)
```

この明示的な分離により、データがどこに保存されるかが明確になり、ストレージコンテキストに関する混乱を排除します。

## 詳細なコアコンセプト

### Experimentオブジェクトの理解

`Experiment`クラスはMINTOへの主要なインターフェースです。以下を管理します：

```python
# 基本的な実験の作成
experiment = minto.Experiment(
    name="optimization_study",           # 実験識別子
    savedir="./my_experiments",         # ストレージディレクトリ
    auto_saving=True,                   # 自動永続化
    collect_environment=True            # 環境メタデータ
)

# 主要なプロパティ
print(f"実験名: {experiment.name}")
print(f"タイムスタンプ: {experiment.timestamp}")
print(f"実行中: {experiment._running}")
print(f"現在のランID: {experiment._run_id}")
```

### データ階層

MINTOは実験レベルとランレベルのデータを明示的に分離します：
- **実験レベルデータ**：すべてのランで共有（問題、インスタンス、グローバルパラメータ）
- **ランレベルデータ**：各ランイテレーションに固有（ソリューション、ランパラメータ）

この明示的な分離は、以前の暗黙的なwith句の動作を置き換え、データがどこに保存されるかをより明確にします。

#### 実験レベルデータ（共有）

```python
# 問題：数学的定式化
import jijmodeling as jm

problem = jm.Problem("traveling_salesman")
n = jm.Placeholder("n")
x = jm.BinaryVar("x", shape=(n, n))
# ... 目的関数と制約を定義 ...

experiment.log_global_problem("tsp", problem)

# インスタンス：具体的な問題データ
cities = load_tsp_cities("berlin52.tsp")
instance = create_tsp_instance(cities)
experiment.log_global_instance("berlin52", instance)

# 設定オブジェクト：専用のconfigメソッドを使用
algorithm_config = {
    "population_size": 100,
    "elite_ratio": 0.1,
    "crossover_operators": ["ox", "pmx"]
}
experiment.log_global_config("genetic_config", algorithm_config)
```

#### ランレベルデータ（イテレーションごと）

```python
# 異なるパラメータで複数のランを実行
temperatures = [100, 500, 1000, 2000]

for temp in temperatures:
    run = experiment.run()  # 明示的なrunオブジェクトを作成
    with run:  # 自動クリーンアップのためのコンテキストマネージャ
        # ラン固有のパラメータをログ
        run.log_parameter("temperature", temp)
        run.log_parameter("cooling_rate", 0.95)
        
        # 解を求めて結果をログ
        solution = simulated_annealing(problem, temp)
        run.log_solution("sa_solution", solution)
        
        # パフォーマンスメトリクスをログ
        run.log_parameter("objective", solution.objective)
        run.log_parameter("feasible", solution.is_feasible)
        run.log_parameter("runtime", solution.elapsed_time)
```

### データ型とストレージ

#### シンプルなパラメータ

MINTOはさまざまなデータ型を自動的に処理します：

```python
# 実験レベルパラメータ（ラン間で共有）
experiment.log_global_parameter("learning_rate", 0.001)      # float
experiment.log_global_parameter("population_size", 100)      # int
experiment.log_global_parameter("algorithm", "genetic")      # str

# コレクション
experiment.log_global_parameter("layer_sizes", [64, 128, 64])           # list
experiment.log_global_parameter("hyperparams", {"lr": 0.01, "decay": 0.9})  # dict

# NumPy配列
import numpy as np
experiment.log_global_parameter("weights", np.array([0.1, 0.5, 0.4]))   # ndarray

# ラン固有のパラメータ（各イテレーションで異なる）
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_parameter("current_loss", 0.234)
    run.log_parameter("batch_size", 32)
```

#### 複雑なオブジェクト

最適化固有のオブジェクトに対して、MINTOは専用のメソッドを提供します：

```python
# 実験レベルオブジェクト（共有データ）
experiment.log_global_problem("knapsack", knapsack_problem)
experiment.log_global_instance("test_case", problem_instance)

# ラン固有オブジェクト（イテレーションごと）
run = experiment.run()
with run:
    run.log_solution("optimal_solution", solution)
    
    # サンプルセット（OMMX/JijModeling）
    run.log_sampleset("samples", sample_collection)
```

## 高度な機能

### 自動ソルバー統合

`log_solver`デコレータはソルバーの動作を自動的にキャプチャします：

```python
# デコレータアプローチ
@experiment.log_solver
def my_genetic_algorithm(problem, population_size=100, generations=1000):
    """
    すべてのパラメータ（population_size、generations）は自動的にログされます。
    問題オブジェクトはキャプチャされ、適切に保存されます。
    戻り値は型に基づいて分析され、ログされます。
    """
    # 実装をここに記述
    return best_solution

# 使用はすべてを自動的にログ
result = my_genetic_algorithm(tsp_problem, population_size=50)

# パラメータ除外を伴う明示的なアプローチ
solver = experiment.log_solver(
    "genetic_algorithm", 
    my_genetic_algorithm,
    exclude_params=["debug_mode"]  # デバッグパラメータをログしない
)
result = solver(tsp_problem, population_size=50, debug_mode=True)
```

### 環境メタデータ

MINTOは包括的な環境情報を自動的にキャプチャします：

```python
# キャプチャされた環境データを表示
experiment.print_environment_summary()
# 出力:
# 環境サマリー:
# OS: macOS 14.1.1
# Python: 3.11.5
# CPU: Apple M2 Pro (12 cores)
# Memory: 32.0 GB
# 仮想環境: /opt/conda/envs/minto
# 主要パッケージ:
#   - minto: 1.0.0
#   - jijmodeling: 1.7.0
#   - ommx: 0.2.0

# 詳細な環境情報にアクセス
env_info = experiment.get_environment_info()
print(env_info["hardware"]["cpu_count"])  # 12
print(env_info["packages"]["numpy"])       # "1.24.3"
```

### データの永続化

#### ディレクトリベースのストレージ

```python
# 自動保存（デフォルト）
experiment = minto.Experiment("study", auto_saving=True)
# 各log_*呼び出し後にデータが保存される

# 手動保存
experiment = minto.Experiment("study", auto_saving=False)
# ... 作業を行う ...
experiment.save()  # 明示的な保存

# カスタム保存場所
experiment.save("/path/to/custom/directory")

# 読み込み
loaded_experiment = minto.Experiment.load_from_dir(
    "/path/to/saved/experiment"
)
```

#### OMMXアーカイブ形式

```python
# ポータブルアーカイブとして保存
artifact = experiment.save_as_ommx_archive("study.ommx")

# アーカイブから読み込み
experiment = minto.Experiment.load_from_ommx_archive("study.ommx")

# GitHub経由で共有（セットアップが必要）
experiment.push_github(
    org="my-organization",
    repo="optimization-studies",
    name="parameter_sweep_v1"
)
```

### データ分析と可視化

#### テーブル生成

```python
# ランレベルの結果テーブル
results = experiment.get_run_table()
print(results.head())
#   run_id algorithm  temperature  objective  solve_time
# 0      0        sa          100      -1234       0.45
# 1      1        sa          500       -987       0.32
# 2      2        sa         1000       -856       0.28

# 実験レベルのテーブル
tables = experiment.get_experiment_tables()
print(tables["problems"].head())      # 問題定義
print(tables["instances"].head())     # インスタンスデータ
print(tables["parameters"].head())    # すべてのパラメータ
```

#### データ分析パターン

```python
import matplotlib.pyplot as plt
import seaborn as sns

# パフォーマンス分析
results = experiment.get_run_table()

# 温度 vs. 目的関数値
plt.figure(figsize=(10, 6))
plt.plot(results["parameter"]["temperature"], 
         results["parameter"]["objective"], 'o-')
plt.xlabel("温度")
plt.ylabel("目的関数値")
plt.title("シミュレーテッドアニーリング：温度感度")
plt.show()

# アルゴリズム比較（複数のアルゴリズムがある場合）
if "algorithm" in results["parameter"].columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, x=("parameter", "algorithm"), 
                y=("parameter", "objective"))
    plt.title("アルゴリズムパフォーマンス比較")
    plt.show()

# 収束分析
convergence_data = results[["parameter"]].copy()
convergence_data["iteration"] = range(len(results))
plt.plot(convergence_data["iteration"], 
         convergence_data[("parameter", "objective")])
plt.xlabel("イテレーション")
plt.ylabel("最良目的関数値")
plt.title("収束履歴")
plt.show()
```

## ベストプラクティスとパターン

### 実験設計

#### 1. 焦点を絞った実験

```python
# 良い例：明確で焦点を絞ったスコープ
experiment_temp = minto.Experiment("sa_temperature_analysis")
experiment_crossover = minto.Experiment("ga_crossover_comparison")

# 避けるべき例：混在した焦点のない実験
experiment_everything = minto.Experiment("all_optimization_work")
```

#### 2. 体系的なパラメータスイープ

```python
# 構造化されたパラメータ探索
experiment = minto.Experiment("hyperparameter_optimization")

# パラメータ空間の定義
param_space = {
    "population_size": [50, 100, 200, 400],
    "crossover_rate": [0.6, 0.7, 0.8, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1, 0.2]
}

# 体系的な探索
from itertools import product

for pop_size, cx_rate, mut_rate in product(*param_space.values()):
    run = experiment.run()
    with run:
        # パラメータの組み合わせをログ
        run.log_parameter("population_size", pop_size)
        run.log_parameter("crossover_rate", cx_rate)
        run.log_parameter("mutation_rate", mut_rate)
        
        # 最適化を実行
        solution = genetic_algorithm(
            problem=tsp_problem,
            population_size=pop_size,
            crossover_rate=cx_rate,
            mutation_rate=mut_rate
        )
        
        # 結果をログ
        run.log_solution("ga_solution", solution)
        run.log_parameter("objective", solution.objective)
        run.log_parameter("generations", solution.generations)
```

#### 3. 包括的なロギング

```python
run = experiment.run()
with run:
    start_time = time.time()
    
    # アルゴリズム設定（ラン固有）
    run.log_parameter("algorithm", "genetic_algorithm")
    run.log_parameter("population_size", 100)
    run.log_parameter("max_generations", 1000)
    
    # 問題の特性（共有の場合は実験レベルに）
    run.log_parameter("problem_size", len(cities))
    run.log_parameter("problem_type", "symmetric_tsp")
    
    # 最適化を実行
    solution = optimize()
    
    # ソリューションの品質
    run.log_parameter("objective_value", solution.objective)
    run.log_parameter("feasible", solution.is_feasible)
    run.log_parameter("optimality_gap", solution.gap)
    
    # パフォーマンスメトリクス
    run.log_parameter("solve_time", time.time() - start_time)
    run.log_parameter("iterations", solution.iterations)
    run.log_parameter("evaluations", solution.function_evaluations)
    
    # ソリューション自体
    run.log_solution("best_solution", solution)
```

### データ整理

#### 命名規則

```python
# 説明的で検索可能な名前を使用
experiment.log_global_parameter("simulated_annealing_temperature", 1000)
experiment.log_global_parameter("genetic_algorithm_population_size", 100)

# ランレベルのソリューションロギング
run = experiment.run()
with run:
    run.log_solution("clarke_wright_routes", cw_solution)

# 関連する場合は単位を含める
experiment.log_global_parameter("time_limit_seconds", 300)
experiment.log_global_parameter("memory_limit_mb", 2048)

# 関連するパラメータには一貫したプレフィックスを使用
experiment.log_global_parameter("sa_temperature", 1000)
experiment.log_global_parameter("sa_cooling_rate", 0.95)
experiment.log_global_parameter("sa_min_temperature", 0.01)
```

#### 実験の比較

```python
# 分析のために関連する実験を結合
experiments = [
    minto.Experiment.load_from_dir("genetic_algorithm_study"),
    minto.Experiment.load_from_dir("simulated_annealing_study"),
    minto.Experiment.load_from_dir("tabu_search_study")
]

# 結合分析を作成
combined = minto.Experiment.concat(
    experiments, 
    name="algorithm_comparison_meta_study"
)

# すべてのアルゴリズムで分析
results = combined.get_run_table()
algorithm_performance = results.groupby(("parameter", "algorithm")).agg({
    ("parameter", "objective"): ["mean", "std", "min", "max"],
    ("parameter", "solve_time"): ["mean", "std"]
})
print(algorithm_performance)
```

## トラブルシューティング

### よくある問題

#### データが永続化されない

```python
# 問題：auto_saving=Falseだが手動保存なし
experiment = minto.Experiment("study", auto_saving=False)
# ... 作業を行う ...
# 明示的な保存がない場合、データは失われる！

# 解決策：auto_savingを有効にするかsave()を呼び出す
experiment = minto.Experiment("study", auto_saving=True)
# または
experiment.save()  # 手動保存
```

#### パラメータ型エラー

```python
# 問題：シリアライズできないオブジェクトをパラメータとして使用
class CustomObject:
    pass

experiment.log_parameter("custom", CustomObject())  # ValueError発生

# 解決策：複雑なデータにはlog_objectを使用
experiment.log_object("custom_config", {
    "type": "CustomObject",
    "parameters": {"value": 42}
})
```

#### ランコンテキストの混乱

```python
# 旧：混乱を招く暗黙的なコンテキスト動作（非推奨）
experiment.log_parameter("global_param", "value")  # 実験レベル

with experiment.run():
    experiment.log_parameter("run_param", "value")  # コンテキストによりランレベル

# 新：明示的で明確な分離
experiment.log_global_parameter("global_param", "value")  # 常に実験レベル

run = experiment.run()
with run:
    run.log_parameter("run_param", "value")  # 明確にランレベル
```

### パフォーマンス最適化

#### 大規模データセット

```python
# 大規模なソリューションオブジェクトの場合、サマリーの保存を検討
run = experiment.run()
with run:
    large_solution = solve_large_problem()
    
    # 完全なソリューションの代わりにサマリーを保存
    run.log_parameter("objective", large_solution.objective)
    run.log_parameter("solution_size", len(large_solution.variables))
    run.log_parameter("nonzero_count", large_solution.nonzeros)
    
    # 選択的に完全なソリューションを保存
    if large_solution.is_optimal:
        run.log_solution("optimal_solution", large_solution)
```

#### バッチ処理

```python
# 効率的なパラメータロギング
parameter_batch = {
    "algorithm": "genetic",
    "population_size": 100,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1
}

run = experiment.run()
with run:
    run.log_params(parameter_batch)  # 複数のパラメータを単一呼び出しで
```

## 統合例

### 人気のある最適化ライブラリとの統合

#### OR-Tools統合

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

experiment = minto.Experiment("ortools_vrp_study")

@experiment.log_solver
def solve_vrp_ortools(distance_matrix, vehicle_count=1, depot=0):
    # OR-Tools VRP ソルビング
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), vehicle_count, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    # コストコールバックを定義し、コスト次元を追加
    # ... OR-Tools固有のコード ...
    
    # 解く
    solution = routing.SolveWithParameters(search_parameters)
    return extract_solution(solution, manager, routing)

# 使用はパラメータと結果を自動的にログ
vrp_solution = solve_vrp_ortools(
    distance_matrix=cities_distances,
    vehicle_count=3,
    depot=0
)
```

#### CPLEX統合（OMMX経由）

```python
import ommx_cplex_adapter as cplex_ad

experiment = minto.Experiment("cplex_milp_study")

# 問題とインスタンスをログ
experiment.log_global_problem("facility_location", milp_problem)
experiment.log_global_instance("northeast_facilities", problem_instance)

time_limits = [60, 300, 1800]  # 1分、5分、30分

for time_limit in time_limits:
    run = experiment.run()
    with run:
        run.log_parameter("time_limit_seconds", time_limit)
        run.log_parameter("solver", "cplex")
        
        # CPLEXで解く
        adapter = cplex_ad.OMMXCPLEXAdapter(problem_instance)
        adapter.set_time_limit(time_limit)
        
        solution = adapter.solve()
        
        run.log_solution("cplex_solution", solution)
        run.log_parameter("objective", solution.objective)
        run.log_parameter("solve_status", str(solution.status))
        run.log_parameter("gap", solution.mip_gap)
```

### Jupyterノートブック統合

```python
# ノートブックフレンドリーな実験管理
%matplotlib inline
import matplotlib.pyplot as plt

# 実験を作成
experiment = minto.Experiment("notebook_optimization_study")

# インタラクティブなパラメータ探索
from ipywidgets import interact, FloatSlider

@interact(temperature=FloatSlider(min=1, max=1000, step=10, value=100))
def run_experiment(temperature):
    run = experiment.run()
    with run:
        run.log_parameter("temperature", temperature)
        
        solution = simulated_annealing(problem, temperature)
        run.log_solution("sa_solution", solution)
        run.log_parameter("objective", solution.objective)
        
        # リアルタイム可視化
        results = experiment.get_run_table()
        plt.figure(figsize=(8, 4))
        plt.plot(results["parameter"]["temperature"], 
                results["parameter"]["objective"], 'o-')
        plt.xlabel("温度")
        plt.ylabel("目的関数値")
        plt.title("SA パフォーマンス vs 温度")
        plt.grid(True)
        plt.show()
        
        print(f"最新の結果: {solution.objective:.2f}")
```

この包括的なユーザーガイドは、実際の最適化研究開発ワークフローでMINTOを効果的に使用するための実践的な知識を提供します。これらのパターンとベストプラクティスに従うことで、ユーザーは堅牢で再現可能な最適化実験を構築できます。