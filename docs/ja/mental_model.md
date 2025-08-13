# MINTOの理解：メンタルモデルと設計思想

## MINTOとは？

MINTOは「最適化のためのMLflow」です。機械学習の実験追跡にMLflowが革命をもたらしたように、MINTOは最適化領域に構造化された実験管理をもたらすPythonフレームワークです。

## コアメンタルモデル

### 2レベルストレージアーキテクチャ

MINTOの中核は、最適化研究が自然に組織化される方法を反映した**2レベルストレージアーキテクチャ**で動作します：

1. **実験レベル** - 共有される再利用可能なデータ：
   - **問題（Problems）**：最適化の目的と制約を定義する数学的定式化
   - **インスタンス（Instances）**：問題をパラメータ化する具体的なデータ（グラフ構造、行列など）
   - **オブジェクト（Objects）**：カスタムデータ構造と設定
   - **環境メタデータ**：再現性のためのシステム情報

2. **ランレベル** - 反復固有のデータ：
   - **パラメータ（Parameters）**：アルゴリズム設定、ハイパーパラメータ、制御値
   - **解（Solutions）**：最適化結果と解ベクトル
   - **サンプルセット（Samplesets）**：解サンプルのコレクション（確率的ソルバー用）
   - **性能メトリクス**：時間、目的関数値、品質指標

この分離は、研究者が問題を一度定義してから、異なるパラメータ、アルゴリズム、または設定で多くのバリエーションを実行する自然なワークフローを反映しています。

### 明示的なラン作成パターン

MINTOは実験全体のデータとラン固有のデータを明確に分離するために、明示的なラン作成を使用します：

```python
import minto

# 実験を作成
experiment = minto.Experiment("my_optimization_study")

# 実験全体のデータをログ（すべてのランで共有）
experiment.log_global_problem("tsp", traveling_salesman_problem)
experiment.log_global_instance("berlin52", berlin_52_instance)

# ラン固有のデータをログ（各実行に固有）
for temperature in [100, 500, 1000]:
    run = experiment.run()  # 新しいランを明示的に作成
    with run:  # 自動クリーンアップのためランコンテキストを使用
        run.log_parameter("temperature", temperature)
        solution = solve_with_simulated_annealing(temperature)
        run.log_solution("sa_result", solution)
```

このパターンにより以下が保証されます：
- **明確な関心事の分離**：実験データとランデータが明示的
- **暗黙的な動作なし**：データがどこに保存されるかが常に明確
- **より良いメンタルモデル**：自然な最適化ワークフローと一致
- 体系的なデータ整理
- ラン間での簡単な分析

## 主要な設計原則

### 1. デフォルトでの再現性

MINTOは以下を含む環境メタデータを自動的にキャプチャします：
- オペレーティングシステムとバージョン
- ハードウェア仕様（CPU、メモリ）
- Pythonバージョンと仮想環境
- 最適化ライブラリのパッケージバージョン
- 実行タイムスタンプ

これにより、異なる環境とシステム間で実験を再現できることが保証されます。

### 2. 柔軟なデータ型

MINTOはシンプルなデータ構造と複雑なデータ構造の両方をサポートします：

**シンプル型**（パラメータとして保存）：
- スカラー：`int`、`float`、`str`
- 基本的なコレクション：`list`、`dict`
- 配列：`numpy.ndarray`

**複雑型**（オブジェクトとして保存）：
- 最適化問題（`jijmodeling.Problem`）
- 問題インスタンス（`ommx.v1.Instance`）
- 解（`ommx.v1.Solution`）
- サンプルセット（`ommx.v1.SampleSet`、`jijmodeling.SampleSet`）

### 3. 複数のストレージ形式

MINTOは実験の永続化方法に柔軟性を提供します：

- **ディレクトリベースのストレージ**：人間が読める、バージョン管理に適した形式
- **OMMXアーカイブ**：標準化された、ポータブルなバイナリ形式
- **GitHub統合**：直接的な共有とコラボレーションのサポート

### 4. 自動ソルバー統合

`log_solver`メソッドは実験レベルとランレベルの両方で使用できます：

```python
# 実験レベルのソルバーログ（問題/インスタンス登録）
@experiment.log_solver
def setup_problem(problem_data):
    # 問題セットアップロジック
    return problem_instance

# ランレベルのソルバーログ（アルゴリズム実行）
run = experiment.run()
with run:
    @run.log_solver
    def my_optimization_solver(problem, temperature=1000, iterations=10000):
        # ソルバー実装
        return solution
    
    # すべてのパラメータと結果がこのランに自動的にログされる
    result = my_optimization_solver(tsp_problem, temperature=500)
```

## データフローの理解

### 実験のライフサイクル

1. **初期化**：環境の自動キャプチャを伴う実験の作成
2. **セットアップ**：問題、インスタンス、共有オブジェクトのログ
3. **実行**：コンテキストマネージャー内での最適化反復の実行
4. **分析**：ログされたデータからのテーブルと視覚化の生成
5. **永続化**：ディスクへの保存またはアーカイブを介した共有

### データの関係性

```text
実験
├── 環境メタデータ（自動）
├── 問題（ラン間で共有）
├── インスタンス（ラン間で共有）
├── オブジェクト（共有設定）
└── ラン
    ├── ラン 0
    │   ├── パラメータ（アルゴリズム設定）
    │   ├── 解（最適化結果）
    │   └── メタデータ（ラン固有の情報）
    ├── ラン 1
    │   ├── パラメータ
    │   ├── 解
    │   └── メタデータ
    └── ...
```

### テーブル生成

MINTOは分析用の構造化されたテーブルを自動的に生成します：

```python
# ランレベルの結果テーブルを取得
results = experiment.get_run_table()
print(results)
#   run_id  temperature  objective_value  solve_time
# 0      0          100            -1234        0.45
# 1      1          500             -987        0.32
# 2      2         1000             -856        0.28

# 詳細な実験情報へのアクセス
tables = experiment.get_experiment_tables()
# 返される：problems、instances、objects、parameters、solutionsテーブル
```

## ベストプラクティスと使用パターン

### 1. 実験の整理

**研究課題ごとに実験を構造化する：**
```python
# 良い例：焦点を絞った実験スコープ
experiment = minto.Experiment("temperature_sensitivity_analysis")

# 避けるべき例：無関係な研究の混在
experiment = minto.Experiment("all_my_optimization_work")
```

**意味のある名前を使用する：**
```python
# 良い例：記述的で検索可能
experiment.log_global_parameter("simulated_annealing_temperature", 1000)
experiment.log_global_solution("best_tour", optimal_solution)

# 避けるべき例：暗号的な略語
experiment.log_global_parameter("sa_temp", 1000)
experiment.log_global_solution("sol", optimal_solution)
```

### 2. データログ戦略

**ストレージレベルによって関心事を分離する：**
```python
# 実験レベル：問題定義と共有データ
experiment.log_global_problem("vehicle_routing", vrp_problem)
experiment.log_global_instance("customer_locations", berlin_customers)

# ランレベル：アルゴリズムパラメータと結果
run = experiment.run()
with run:
    run.log_parameter("vehicle_capacity", 100)
    run.log_parameter("algorithm", "clarke_wright")
    run.log_solution("routes", best_routes)
```

**包括的なメタデータをログする：**
```python
run = experiment.run()
with run:
    # アルゴリズム設定
    run.log_parameter("population_size", 100)
    run.log_parameter("crossover_rate", 0.8)
    run.log_parameter("mutation_rate", 0.1)
    
    # 性能メトリクス
    run.log_parameter("objective_value", solution.objective)
    run.log_parameter("solve_time", elapsed_time)
    run.log_parameter("iterations", total_iterations)
    
    # 解の品質
    run.log_parameter("feasible", solution.is_feasible)
    run.log_parameter("optimality_gap", gap_percentage)
```

### 3. ソルバー統合

**自動キャプチャのためにlog_solverデコレータを使用する：**
```python
# 自動的なパラメータと結果のログ
@run.log_solver
def genetic_algorithm(problem, population_size=100, generations=1000):
    # 実装
    return best_solution

# または明示的なソルバーログ
solver = run.log_solver("genetic_algorithm", genetic_algorithm)
result = solver(tsp_problem, population_size=50)
```

**複雑なソルバー出力を処理する：**
```python
run = experiment.run()
with run:
    result = complex_solver(problem)
    
    # 複数の解コンポーネントをログ
    run.log_solution("primary_solution", result.best_solution)
    run.log_parameter("convergence_history", result.objective_history)
    run.log_parameter("computation_stats", result.statistics)
```

### 4. 分析と視覚化

**比較分析を生成する：**
```python
# 実験をロードして結合
experiments = [
    minto.Experiment.load_from_dir("exp_genetic_algorithm"),
    minto.Experiment.load_from_dir("exp_simulated_annealing"),
    minto.Experiment.load_from_dir("exp_tabu_search")
]

combined = minto.Experiment.concat(experiments, name="algorithm_comparison")
results = combined.get_run_table()

# アルゴリズム間の性能を分析
import matplotlib.pyplot as plt
results.groupby("algorithm")["objective_value"].mean().plot(kind="bar")
plt.title("アルゴリズム性能比較")
plt.show()
```

## 一般的な使用パターン

### 1. パラメータスイープ

```python
experiment = minto.Experiment("parameter_sensitivity")
experiment.log_global_problem("quadratic_assignment", qap_problem)

for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        run = experiment.run()
        with run:
            run.log_parameter("penalty_alpha", alpha)
            run.log_parameter("penalty_beta", beta)
            
            solution = solve_with_penalties(alpha, beta)
            run.log_solution("penalized_solution", solution)
            run.log_parameter("objective", solution.objective)
```

### 2. アルゴリズムベンチマーキング

```python
experiment = minto.Experiment("solver_benchmark")

# 標準ベンチマークインスタンスをロード
for instance_name in ["kroA100", "kroB100", "kroC100"]:
    instance = load_tsplib_instance(instance_name)
    experiment.log_global_instance(instance_name, instance)

algorithms = ["genetic", "simulated_annealing", "ant_colony"]

for algorithm in algorithms:
    for instance_name in experiment.dataspace.experiment_datastore.instances:
        run = experiment.run()
        with run:
            run.log_parameter("algorithm", algorithm)
            run.log_parameter("instance", instance_name)
            
            solver = get_solver(algorithm)
            solution = solver.solve(instance)
            
            run.log_solution("result", solution)
            run.log_parameter("objective", solution.objective)
            run.log_parameter("solve_time", solution.runtime)
```

### 3. ハイパーパラメータ最適化

```python
experiment = minto.Experiment("hyperparameter_tuning")
experiment.log_global_problem("scheduling", job_shop_problem)

from itertools import product

# パラメータグリッドを定義
param_grid = {
    "population_size": [50, 100, 200],
    "crossover_rate": [0.7, 0.8, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1]
}

# グリッドサーチ
for params in product(*param_grid.values()):
    pop_size, crossover, mutation = params
    
    run = experiment.run()
    with run:
        run.log_parameter("population_size", pop_size)
        run.log_parameter("crossover_rate", crossover)
        run.log_parameter("mutation_rate", mutation)
        
        solution = genetic_algorithm(
            problem=job_shop_problem,
            population_size=pop_size,
            crossover_rate=crossover,
            mutation_rate=mutation
        )
        
        run.log_solution("optimized_schedule", solution)
        run.log_parameter("makespan", solution.makespan)
        run.log_parameter("tardiness", solution.total_tardiness)
```

## 最適化エコシステムとの統合

### OMMX互換性

MINTOはOpen Mathematical Modeling Exchange（OMMX）標準を中心に構築されています：

- **ネイティブOMMXサポート**：問題、インスタンス、解がシームレスに動作
- **標準化されたフォーマット**：ツール間の相互運用性を保証
- **アーカイブ互換性**：OMMXアーカイブとの直接インポート/エクスポート

### JijModeling統合

MINTOはJijModeling問題に対する第一級のサポートを提供します：

```python
import jijmodeling as jm

# 最適化問題を定義
problem = jm.Problem("knapsack")
x = jm.BinaryVar("x", shape=(n,))
problem += jm.sum(i, values[i] * x[i])  # 目的関数
problem += jm.Constraint("capacity", jm.sum(i, weights[i] * x[i]) <= capacity)

# 自動変換とログ
experiment.log_global_problem("knapsack", problem)
```

### ソルバーエコシステム

MINTOはさまざまな最適化ソルバーと連携します：

- **商用ソルバー**：CPLEX、Gurobi（OMMXアダプター経由）
- **オープンソースソルバー**：SCIP、OR-Tools（OMMXアダプター経由）
- **メタヒューリスティクス**：OpenJij、カスタム実装
- **クラウドソルバー**：JijZept、D-Waveシステム

## まとめ

MINTOのメンタルモデルは、体系的で再現可能な最適化研究を中心としています。2レベルストレージアーキテクチャ、コンテキストマネージャーパターン、自動メタデータキャプチャを理解することで、ユーザーはMINTOを活用して以下を実現できます：

- **研究ワークフローの合理化**：定型的なコードを削減し、最適化問題に集中
- **再現性の保証**：自動環境キャプチャと体系的なデータ整理
- **コラボレーションの実現**：標準化されたフォーマットと共有メカニズム
- **発見の加速**：構造化された分析と視覚化ツール

このフレームワークは、多様な最適化のユースケースに対する柔軟性を維持しながら、実験管理の複雑さを抽象化します。学術研究、産業最適化、アルゴリズム開発のいずれを行う場合でも、MINTOは厳密で体系的な実験のための基盤を提供します。