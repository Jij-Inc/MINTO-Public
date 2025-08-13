# クイックスタート

このガイドでは、MINTOを活用するための初期ステップを説明します。このセクションでは、MINTOのセットアップから実験データの処理まで、4つの主要なステップをカバーします。

- MINTOのインストール方法
- 実験データの記録
- 記録した実験データのテーブル表示
- 実験データの保存と読み込み

## インストール

```bash
pip install minto
```

## 実験データの記録

`minto`を使用すると、実験中に生成されるさまざまなデータを簡単に記録できます。  
以下の例では、OMMXでサポートされているPySCIPOptを使用して、MIPソルバーの時間制限依存性に関する数値実験を行います。
`minto`はOMMX Messageをネイティブにサポートしているため、OMMXを通じてスムーズに数値実験を実行できます。

試してみましょう。

```python
import minto
import ommx_pyscipopt_adapter as scip_ad
from ommx.dataset import miplib2017
```

このチュートリアルでは、ベンチマーク対象としてmiplib2017からインスタンスを選択します。ommx.datasetを使用して、miplib2017インスタンスを簡単に取得できます。

```python
instance_name = "reblock115"
instance = miplib2017(instance_name)
```

`ommx_pyscipopt_adapter`を使用して、ommx.v1.InstanceをPySCIPOptのModelに変換し、limits/timeパラメータを変更して実験を行います。

ommxのインスタンスとソリューションは、MINTOの`.log_*`メソッドを使用して保存できます。この数値実験では単一のインスタンスを使用し、変更されないため、`log_global_instance`を使用して実験レベルで保存します。
ソリューションは各時間制限ごとに異なるため、明示的な`run`オブジェクトを使用してランレベルで保存します。

```python
timelimit_list = [0.1, 0.5, 1, 2]

# auto_savingがTrueの場合、Experiment.log_*メソッドが呼ばれるたびにデータが自動的に保存されます。
# データは段階的に保存されるため、途中でエラーが発生しても、その時点までのデータは失われません。
experiment = minto.Experiment(
    name='scip_exp',
    auto_saving=True
)

# 実験レベルのデータにはlog_global_instanceを使用
experiment.log_global_instance(instance_name, instance)
adapter = scip_ad.OMMXPySCIPOptAdapter(instance)
scip_model = adapter.solver_input

for timelimit in timelimit_list:
    # 明示的なrunオブジェクトを作成
    run = experiment.run()
    with run:
        run.log_parameter("timelimit", timelimit)

        # SCIPで解く
        scip_model.setParam("limits/time", timelimit)
        scip_model.optimize()
        solution = adapter.decode(scip_model)

        run.log_solution("scip", solution)
```

`.get_run_table`メソッドを使用してommx.Solutionをpandas.DataFrameに変換すると、ソリューションの主要な情報のみが表示されます。実際のソリューションオブジェクトにアクセスしたい場合は、`experiment.dataspaces.run_datastores[run_id].solutions`から参照できます。

```python
runs_table = experiment.get_run_table()
runs_table
```

## 実験データの保存と読み込み

`Experiment.save`メソッドを使用して、いつでもデータを保存できます。  
`Experiment.load_from_dir`メソッドを使用して、保存されたデータを読み込むこともできます。

```python
# デフォルトでは、データは.minto_experiments/ディレクトリに保存されます
experiment.save()
```

```python
exp2 = minto.Experiment.load_from_dir(
    ".minto_experiments/" + experiment.experiment_name
)
```

## OMMXアーカイブへの保存と読み込み

`Experiment.save_to_ommx_archive`および`Experiment.load_from_ommx_archive`メソッドを使用して、OMMXアーカイブ形式でデータを保存および読み込むことができます。

```python
artifact = experiment.save_as_ommx_archive("scip_experiment.ommx")
```

```python
exp3 = minto.Experiment.load_from_ommx_archive("scip_experiment.ommx")
exp3.get_run_table()
```

# まとめ

このチュートリアルでは、`minto`を使用して数値実験を管理する方法を学びました：

1. **インストール**はpipを使用して簡単に行えます：
    ```bash
    pip install minto
    ```

2. **実験データの記録**
    - OMMXインターフェースを通じてSCIPソルバーを使用
    - 異なる時間制限（0.1秒から2.0秒）で実験を実施
    - インスタンスデータとソリューション結果を体系的に記録

3. **データ管理**
    - `auto_saving=True`で自動保存
    - `experiment.save()`で手動保存
    - pandas DataFrameとmatplotlibを使用したデータの可視化
    - 結果は目的関数値が-3.03e+07に収束することを示した

4. **データの永続化**
    - `.minto_experiments/`ディレクトリへのローカル保存
    - OMMXアーカイブ形式のサポート
    - `load_from_dir`と`load_from_ommx_archive`による簡単なデータ復元

このワークフローは、最適化タスクにおける構造化された実験管理のための`minto`の機能を示しています。