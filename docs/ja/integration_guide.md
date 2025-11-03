# MINTO統合・移行ガイド

## 他の実験追跡ツールからの移行

### MLflowからの移行

機械学習実験でMLflowを使用していた方にとって、MINTOは最適化問題に特化されているものの、馴染みのある感覚で使用できます。

#### 概念のマッピング

| MLflow | MINTO | 備考 |
|--------|-------|------|
| `mlflow.start_run()` | `experiment.run()` | 明示的なラン作成 |
| `mlflow.log_param()` | `run.log_parameter()` | ランレベルのパラメータ |
| `mlflow.log_metric()` | `run.log_parameter()` | 性能メトリクス |
| `mlflow.log_artifact()` | `run.log_solution()` | 複雑なオブジェクト |
| - | `experiment.log_global_problem()` | 最適化固有 |
| - | `experiment.log_global_instance()` | 最適化固有 |
| Tracking Server | Directory/OMMX | ローカルまたはアーカイブストレージ |

#### コード移行例

**MLflowスタイル：**
```python
import mlflow

mlflow.set_experiment("optimization_study")

with mlflow.start_run():
    mlflow.log_param("algorithm", "genetic")
    mlflow.log_param("population_size", 100)
    
    solution = optimize()
    
    mlflow.log_metric("objective", solution.objective)
    mlflow.log_metric("runtime", solution.time)
    mlflow.log_artifact("solution.json")
```

**MINTOスタイル：**
```python
import minto

experiment = minto.Experiment("optimization_study")

run = experiment.run()
with run:
    run.log_parameter("algorithm", "genetic")
    run.log_parameter("population_size", 100)
    
    solution = optimize()
    
    run.log_parameter("objective", solution.objective)
    run.log_parameter("runtime", solution.time)
    run.log_solution("optimal_solution", solution)
```

### 手動実験追跡からの移行

#### 移行前：手動CSV/Excel追跡

```python
# 古いアプローチ：手動追跡
import pandas as pd
import csv

results = []

for temperature in [100, 500, 1000]:
    for alpha in [0.1, 0.5, 1.0]:
        solution = simulated_annealing(problem, temperature, alpha)
        
        # 手動で結果を収集
        results.append({
            "temperature": temperature,
            "alpha": alpha,
            "objective": solution.objective,
            "runtime": solution.time,
            "feasible": solution.is_feasible
        })

# 手動で保存
df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)
```

#### 移行後：MINTOアプローチ

```python
# MINTOアプローチ：自動追跡
import minto

experiment = minto.Experiment("sa_parameter_study")
experiment.log_global_problem("tsp", tsp_problem)

for temperature in [100, 500, 1000]:
    for alpha in [0.1, 0.5, 1.0]:
        run = experiment.run()
        with run:
            run.log_parameter("temperature", temperature)
            run.log_parameter("alpha", alpha)
            
            solution = simulated_annealing(problem, temperature, alpha)
            
            run.log_solution("sa_solution", solution)
            run.log_parameter("objective", solution.objective)
            run.log_parameter("runtime", solution.time)
            run.log_parameter("feasible", solution.is_feasible)

# 自動分析
results = experiment.get_run_table()
print(results.head())

# 自動保存（auto_saving=Trueの場合）
experiment.save()
```

## 既存ワークフローへのMINTO統合

### 研究室での統合

#### 共有実験のセットアップ

```python
# 研究室実験用の共有ベースディレクトリ
import pathlib

LAB_EXPERIMENTS_DIR = pathlib.Path("/shared/optimization_lab/experiments")

def create_lab_experiment(researcher_name, study_name):
    """研究室メンバー用の標準実験作成。"""
    experiment_name = f"{researcher_name}_{study_name}_{datetime.now().strftime('%Y%m%d')}"
    
    experiment = minto.Experiment(
        name=experiment_name,
        savedir=LAB_EXPERIMENTS_DIR,
        auto_saving=True,
        collect_environment=True
    )
    
    # 研究者情報をログ
    experiment.log_global_parameter("researcher", researcher_name)
    experiment.log_global_parameter("institution", "最適化研究室")
    
    return experiment

# 研究室メンバーによる使用
experiment = create_lab_experiment("alice", "genetic_algorithm_study")
```

#### 標準化問題ライブラリ

```python
# 研究室問題レジストリ
class LabProblemRegistry:
    def __init__(self, base_dir):
        self.base_dir = pathlib.Path(base_dir)
        self.problems = {}
        self.instances = {}
        
    def register_problem(self, name, problem, description=""):
        """研究室で使用する標準問題を登録。"""
        self.problems[name] = {
            "problem": problem,
            "description": description,
            "added_by": "lab_admin",
            "added_date": datetime.now()
        }
        
        # 共有場所に保存
        problem_path = self.base_dir / "problems" / f"{name}.pkl"
        problem_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(problem_path, "wb") as f:
            pickle.dump(problem, f)
    
    def get_problem(self, name):
        """標準問題を取得。"""
        if name in self.problems:
            return self.problems[name]["problem"]
        
        # メモリにない場合はディスクから読み込み
        problem_path = self.base_dir / "problems" / f"{name}.pkl"
        if problem_path.exists():
            with open(problem_path, "rb") as f:
                return pickle.load(f)
        
        raise ValueError(f"問題 {name} が見つかりません")

# 研究室での使用
registry = LabProblemRegistry("/shared/optimization_lab/registry")

# すべての研究者が利用可能な標準問題
experiment = create_lab_experiment("bob", "metaheuristics_comparison")
experiment.log_global_problem("tsp_50", registry.get_problem("tsp_50_cities"))
experiment.log_global_problem("vrp_100", registry.get_problem("vrp_100_customers"))
```

### CI/CD統合

#### 自動ベンチマークパイプライン

```python
# benchmark_pipeline.py
import minto
import argparse
import json
from pathlib import Path

def run_benchmark_suite():
    """CI/CD用の自動ベンチマーク実行。"""
    
    # タイムスタンプベースの実験を作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = minto.Experiment(
        name=f"benchmark_suite_{timestamp}",
        savedir=Path("./benchmark_results"),
        auto_saving=True
    )
    
    # 標準ベンチマーク問題をロード
    benchmark_problems = load_benchmark_suite()
    
    for problem_name, problem_data in benchmark_problems.items():
        experiment.log_global_problem(problem_name, problem_data["problem"])
        experiment.log_global_instance(f"{problem_name}_instance", problem_data["instance"])
        
        # スイート内のすべてのアルゴリズムを実行
        algorithms = ["genetic", "simulated_annealing", "tabu_search"]
        
        for algorithm in algorithms:
            run = experiment.run()
            with run:
                run.log_parameter("problem", problem_name)
                run.log_parameter("algorithm", algorithm)
                run.log_parameter("ci_commit", get_git_commit_hash())
                
                solver = get_algorithm_solver(algorithm)
                solution = solver.solve(problem_data["instance"])
                
                run.log_solution(f"{algorithm}_solution", solution)
                run.log_parameter("objective", solution.objective)
                run.log_parameter("runtime", solution.runtime)
                run.log_parameter("feasible", solution.is_feasible)
    
    # 性能レポートを生成
    results = experiment.get_run_table()
    
    # ベースライン（以前の実行）と比較
    baseline_results = load_baseline_results()
    performance_comparison = compare_with_baseline(results, baseline_results)
    
    # 比較レポートを保存
    with open("benchmark_report.json", "w") as f:
        json.dump(performance_comparison, f, indent=2)
    
    # CIの成功/失敗を返す
    return performance_comparison["overall_status"] == "PASS"

if __name__ == "__main__":
    success = run_benchmark_suite()
    exit(0 if success else 1)
```

#### GitHub Actions統合

```yaml
# .github/workflows/optimization_benchmark.yml
name: 最適化アルゴリズムベンチマーク

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 0'  # 週次ベンチマーク

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Pythonセットアップ
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: 依存関係インストール
      run: |
        pip install minto
        pip install -r requirements.txt
    
    - name: ベンチマークスイート実行
      run: |
        python benchmark_pipeline.py
        
    - name: ベンチマーク結果アップロード
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: |
          benchmark_results/
          benchmark_report.json
    
    - name: PR結果コメント
      if: github.event_name == 'pull_request'
      run: |
        python scripts/post_benchmark_comment.py \
          --results benchmark_report.json \
          --pr-number ${{ github.event.number }}
```

### Docker統合

#### コンテナ化された実験

```dockerfile
# Dockerfile.optimization
FROM python:3.11-slim

# 最適化ソルバーインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    coinor-cbc \
    coinor-clp \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MINTOインストール
RUN pip install minto

# 実験コードコピー
COPY experiments/ /experiments/
WORKDIR /experiments

# デフォルトコマンド
CMD ["python", "run_experiment.py"]
```

```python
# run_experiment.py（コンテナ化された実験）
import minto
import os
import sys

def main():
    # 環境変数から設定を取得
    experiment_name = os.getenv("EXPERIMENT_NAME", "containerized_study")
    problem_type = os.getenv("PROBLEM_TYPE", "tsp")
    algorithm = os.getenv("ALGORITHM", "genetic")
    
    # コンテナメタデータ付きで実験を作成
    experiment = minto.Experiment(
        name=f"{experiment_name}_{algorithm}",
        savedir="/results",  # 結果用マウントポイント
        auto_saving=True
    )
    
    # コンテナ情報をログ
    experiment.log_global_parameter("container_id", os.getenv("HOSTNAME", "unknown"))
    experiment.log_global_parameter("docker_image", os.getenv("DOCKER_IMAGE", "unknown"))
    
    # タイプに基づいて問題をロード
    problem, instance = load_problem(problem_type)
    experiment.log_global_problem(problem_type, problem)
    experiment.log_global_instance(f"{problem_type}_instance", instance)
    
    # 最適化を実行
    run = experiment.run()
    with run:
        run.log_parameter("algorithm", algorithm)
        
        solver = get_solver(algorithm)
        solution = solver.solve(instance)
        
        run.log_solution("result", solution)
        run.log_parameter("objective", solution.objective)
        run.log_parameter("runtime", solution.runtime)
    
    # 結果を保存
    experiment.save()
    print(f"実験 {experiment.name} が正常に完了しました")

if __name__ == "__main__":
    main()
```

```bash
# コンテナ化された実験を実行
docker run -v $(pwd)/results:/results \
           -e EXPERIMENT_NAME="docker_optimization" \
           -e PROBLEM_TYPE="vrp" \
           -e ALGORITHM="genetic" \
           optimization:latest
```

### クラウド統合

#### AWS S3ストレージ

```python
import boto3
import minto
from pathlib import Path

class S3ExperimentManager:
    def __init__(self, bucket_name, region="us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region)
    
    def save_experiment_to_s3(self, experiment, key_prefix="experiments/"):
        """MINTO実験をS3に保存。"""
        
        # まずローカルでOMMXアーカイブとして保存
        local_path = f"/tmp/{experiment.name}.ommx"
        artifact = experiment.save_as_ommx_archive(local_path)
        
        # S3にアップロード
        s3_key = f"{key_prefix}{experiment.name}.ommx"
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        
        # ローカルファイルをクリーンアップ
        Path(local_path).unlink()
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def load_experiment_from_s3(self, experiment_name, key_prefix="experiments/"):
        """S3からMINTO実験をロード。"""
        
        # S3からダウンロード
        s3_key = f"{key_prefix}{experiment_name}.ommx"
        local_path = f"/tmp/{experiment_name}.ommx"
        
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
        
        # 実験をロード
        experiment = minto.Experiment.load_from_ommx_archive(local_path)
        
        # ローカルファイルをクリーンアップ
        Path(local_path).unlink()
        
        return experiment

# 使用例
s3_manager = S3ExperimentManager("my-optimization-experiments")

# 実験をクラウドに保存
experiment = minto.Experiment("cloud_optimization_study")
# ... 実験を実行 ...
s3_url = s3_manager.save_experiment_to_s3(experiment)
print(f"実験保存先: {s3_url}")

# クラウドから実験をロード
loaded_experiment = s3_manager.load_experiment_from_s3("cloud_optimization_study")
results = loaded_experiment.get_run_table()
```

#### Azure Blobストレージ

```python
from azure.storage.blob import BlobServiceClient
import minto

class AzureBlobExperimentManager:
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
    
    def save_experiment_to_blob(self, experiment, blob_prefix="experiments/"):
        """MINTO実験をAzure Blobストレージに保存。"""
        
        # OMMXアーカイブとして保存
        local_path = f"/tmp/{experiment.name}.ommx"
        experiment.save_as_ommx_archive(local_path)
        
        # blobストレージにアップロード
        blob_name = f"{blob_prefix}{experiment.name}.ommx"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # クリーンアップ
        Path(local_path).unlink()
        
        return f"https://{blob_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    def load_experiment_from_blob(self, experiment_name, blob_prefix="experiments/"):
        """Azure Blobストレージから MINTO実験をロード。"""
        
        blob_name = f"{blob_prefix}{experiment_name}.ommx"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        # ローカル一時ファイルにダウンロード
        local_path = f"/tmp/{experiment_name}.ommx"
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        # 実験をロード
        experiment = minto.Experiment.load_from_ommx_archive(local_path)
        
        # クリーンアップ
        Path(local_path).unlink()
        
        return experiment
```

## 高度な統合パターン

### 分散コンピューティング統合

#### Dask統合

```python
import minto
import dask
from dask.distributed import Client, as_completed
from dask import delayed

def parallel_optimization_study():
    """Daskを使用した並列最適化実験の実行。"""
    
    # Daskクラスタに接続
    client = Client("scheduler-address:8786")
    
    # ベース実験を作成
    base_experiment = minto.Experiment("parallel_optimization_study")
    
    # パラメータ空間を定義
    parameter_combinations = [
        {"algorithm": "genetic", "pop_size": 50, "generations": 100},
        {"algorithm": "genetic", "pop_size": 100, "generations": 100},
        {"algorithm": "sa", "temperature": 1000, "cooling": 0.95},
        {"algorithm": "sa", "temperature": 2000, "cooling": 0.99},
        # ... さらに多くの組み合わせ
    ]
    
    @delayed
    def run_single_experiment(params):
        """単一の最適化実験を実行。"""
        
        # ワーカー固有の実験を作成
        worker_experiment = minto.Experiment(
            name=f"worker_{params['algorithm']}_{hash(str(params))}",
            savedir=f"/shared/experiments/worker_results",
            auto_saving=True
        )
        
        run = worker_experiment.run()
        with run:
            # パラメータをログ
            for key, value in params.items():
                run.log_parameter(key, value)
            
            # 最適化を実行
            solution = optimize_with_params(params)
            
            # 結果をログ
            run.log_solution("result", solution)
            run.log_parameter("objective", solution.objective)
            run.log_parameter("runtime", solution.runtime)
        
        return worker_experiment.name
    
    # すべての実験を送信
    futures = [run_single_experiment(params) for params in parameter_combinations]
    
    # 完了した結果を収集
    completed_experiments = []
    for future in as_completed(futures):
        experiment_name = future.result()
        completed_experiments.append(experiment_name)
        print(f"完了: {experiment_name}")
    
    # すべてのワーカー実験を結合
    worker_experiments = [
        minto.Experiment.load_from_dir(f"/shared/experiments/worker_results/{name}")
        for name in completed_experiments
    ]
    
    combined_experiment = minto.Experiment.concat(
        worker_experiments,
        name="parallel_optimization_combined"
    )
    
    return combined_experiment

# 使用例
combined_results = parallel_optimization_study()
print(combined_results.get_run_table())
```

#### Ray統合

```python
import ray
import minto

@ray.remote
class OptimizationWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.experiment = minto.Experiment(
            name=f"ray_worker_{worker_id}",
            auto_saving=True
        )
    
    def run_optimization(self, problem, algorithm_params):
        """このワーカーで最適化を実行。"""
        
        run = self.experiment.run()
        with run:
            # ワーカーとパラメータをログ
            run.log_parameter("worker_id", self.worker_id)
            
            for key, value in algorithm_params.items():
                run.log_parameter(key, value)
            
            # 最適化を実行
            solution = solve_problem(problem, algorithm_params)
            
            # 結果をログ
            run.log_solution("worker_solution", solution)
            run.log_parameter("objective", solution.objective)
        
        return {
            "worker_id": self.worker_id,
            "objective": solution.objective,
            "experiment_name": self.experiment.name
        }

def distributed_hyperparameter_search():
    """Rayを使用した分散ハイパーパラメータ探索。"""
    
    # Rayを初期化
    ray.init()
    
    # ワーカープールを作成
    num_workers = 4
    workers = [OptimizationWorker.remote(i) for i in range(num_workers)]
    
    # 探索空間を定義
    search_space = generate_parameter_combinations()
    
    # 作業を分散
    futures = []
    for i, params in enumerate(search_space):
        worker = workers[i % num_workers]
        future = worker.run_optimization.remote(tsp_problem, params)
        futures.append(future)
    
    # 結果を収集
    results = ray.get(futures)
    
    # 最良の結果を見つける
    best_result = min(results, key=lambda x: x["objective"])
    
    # 詳細な分析のため最良の実験をロード
    best_experiment = minto.Experiment.load_from_dir(
        f"./ray_worker_{best_result['worker_id']}"
    )
    
    ray.shutdown()
    
    return best_result, best_experiment
```

### データベース統合

#### PostgreSQL統合

```python
import psycopg2
import pandas as pd
import minto
from sqlalchemy import create_engine

class PostgreSQLExperimentTracker:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.setup_tables()
    
    def setup_tables(self):
        """実験追跡用のテーブルを作成。"""
        
        create_experiments_table = """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            environment_info JSONB,
            description TEXT
        );
        """
        
        runs_table = """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id SERIAL PRIMARY KEY,
            experiment_id INTEGER REFERENCES experiments(experiment_id),
            run_index INTEGER NOT NULL,
            parameters JSONB NOT NULL,
            results JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.engine.connect() as conn:
            conn.execute(create_experiments_table)
            conn.execute(runs_table)
            conn.commit()
    
    def sync_minto_experiment(self, experiment):
        """MINTO実験をPostgreSQLに同期。"""
        
        # 実験レコードを挿入または更新
        experiment_data = {
            "name": experiment.name,
            "environment_info": experiment.get_environment_info(),
            "description": f"{len(experiment.runs)}回の実行を含むMINTO実験"
        }
        
        # 実験を挿入
        insert_experiment_sql = """
        INSERT INTO experiments (name, environment_info, description)
        VALUES (%(name)s, %(environment_info)s, %(description)s)
        ON CONFLICT (name) DO UPDATE SET
            environment_info = EXCLUDED.environment_info,
            description = EXCLUDED.description
        RETURNING experiment_id;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(insert_experiment_sql, experiment_data)
            experiment_id = result.fetchone()[0]
        
        # ランを挿入
        runs_table = experiment.get_run_table()
        
        for idx, run in runs_table.iterrows():
            run_data = {
                "experiment_id": experiment_id,
                "run_index": run["run_id"],
                "parameters": run["parameter"].to_dict(),
                "results": {
                    col: run[col].to_dict() if hasattr(run[col], "to_dict") else str(run[col])
                    for col in runs_table.columns if col != "parameter"
                }
            }
            
            insert_run_sql = """
            INSERT INTO experiment_runs (experiment_id, run_index, parameters, results)
            VALUES (%(experiment_id)s, %(run_index)s, %(parameters)s, %(results)s)
            ON CONFLICT DO NOTHING;
            """
            
            with self.engine.connect() as conn:
                conn.execute(insert_run_sql, run_data)
                conn.commit()
    
    def query_experiments(self, algorithm=None, min_objective=None):
        """データベースから実験をクエリ。"""
        
        query = """
        SELECT e.name, e.created_at, COUNT(r.run_id) as num_runs,
               AVG((r.results->>'objective')::float) as avg_objective
        FROM experiments e
        LEFT JOIN experiment_runs r ON e.experiment_id = r.experiment_id
        WHERE 1=1
        """
        
        params = {}
        
        if algorithm:
            query += " AND r.parameters->>'algorithm' = %(algorithm)s"
            params["algorithm"] = algorithm
            
        if min_objective:
            query += " AND (r.results->>'objective')::float >= %(min_objective)s"
            params["min_objective"] = min_objective
        
        query += " GROUP BY e.experiment_id, e.name, e.created_at ORDER BY e.created_at DESC"
        
        return pd.read_sql(query, self.engine, params=params)

# 使用例
db_tracker = PostgreSQLExperimentTracker("postgresql://user:pass@localhost/experiments")

# MINTO実験をデータベースに同期
experiment = minto.Experiment("database_tracked_study")
# ... 実験を実行 ...
db_tracker.sync_minto_experiment(experiment)

# 実験をクエリ
genetic_experiments = db_tracker.query_experiments(algorithm="genetic")
print(genetic_experiments)
```

この包括的な統合ガイドは、個々の研究プロジェクトから大規模な分散コンピューティング環境まで、既存の研究開発ワークフローにMINTOを組み込むための実践的なパターンを提供します。