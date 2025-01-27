import ommx.v1 as ommx_v1
import pandas as pd

from .v1.datastore import DataStore


def create_table_from_stores(datastores: list[DataStore]):
    df_list = []
    for i, ds in enumerate(datastores):
        records = []
        table_info = create_table_info(ds)
        for storage_name, obj in table_info.items():
            if storage_name in ("parameter", "metadata"):
                v = {(storage_name, k): v for k, v in obj.items()}
                records.append(pd.DataFrame(v, index=[i]))
            else:
                for name, values in obj.items():
                    header = storage_name + "_" + name
                    v = {(header, k): v for k, v in values.items()}
                    v[(header, "name")] = name
                    records.append(pd.DataFrame(v, index=[i]))
        df_list.append(pd.concat(records, axis=1))
    if len(df_list) == 0:
        return pd.DataFrame()
    return pd.concat(df_list)


def create_table(datastore: DataStore):
    table_info = create_table_info(datastore)
    dataframes = {}
    for storage_name, obj in table_info.items():
        records = []
        if storage_name in ("parameter", "metadata"):
            records.append(pd.Series(obj, name=storage_name))
        else:
            for name, values in obj.items():
                records.append(pd.Series(values, name=name))
        df = pd.DataFrame(records)
        dataframes[storage_name] = df

    return dataframes


def create_table_info(datastore: DataStore) -> dict:
    instance_data = {}
    for name, inst in datastore.instances.items():
        instance_data[name] = _extract_instance_info(inst)

    solution_data = {}
    for name, sol in datastore.solutions.items():
        solution_data[name] = _extract_solution_info(sol)

    sampleset_data = {}
    for name, sampleset in datastore.samplesets.items():
        sampleset_data[name] = _extract_sampleset_info(sampleset)

    return {
        "instance": instance_data,
        "solution": solution_data,
        "sampleset": sampleset_data,
        "parameter": datastore.parameters,
        # "objects": datastore.objects,
        "metadata": datastore.meta_data,
    }


def _extract_instance_info(instance: ommx_v1.Instance):
    deci_var_df = instance.decision_variables
    kind_counts = deci_var_df.kind.value_counts()
    info = {
        "num_vars": len(deci_var_df),
        "num_binary": kind_counts.get("binary", 0),
        "num_integer": kind_counts.get("integer", 0),
        "num_continuous": kind_counts.get("continuous", 0),
        "num_cons": instance.num_constraints,
        "title": instance.title,
    }
    return info


def _extract_solution_info(solution: ommx_v1.Solution):
    info = {
        "objective": solution.objective,
        "feasible": solution.feasible,
        "optimality": solution.optimality,
        "relaxation": solution.relaxation,
        "start": solution.start,
    }
    return info


def _extract_sampleset_info(sampleset: ommx_v1.SampleSet):
    summary = sampleset.summary
    objective = summary.objective
    return {
        "num_samples": len(summary),
        "obj_mean": objective.mean(),
        "obj_std": objective.std(),
        "obj_min": objective.min(),
        "obj_max": objective.max(),
        "feasible": summary.feasible.sum(),
        "feasible_unrelaxed": summary.feasible_unrelaxed.sum(),
    }
