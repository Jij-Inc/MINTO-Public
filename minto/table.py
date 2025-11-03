import warnings

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
                if obj:  # Only create DataFrame if obj is not empty
                    v = {(storage_name, k): v for k, v in obj.items()}
                    records.append(pd.DataFrame(v, index=[i]))
            else:
                for name, values in obj.items():
                    if values:  # Only create DataFrame if values is not empty
                        header = storage_name + "_" + name
                        v = {(header, k): v for k, v in values.items()}
                        v[(header, "name")] = name
                        records.append(pd.DataFrame(v, index=[i]))
        if records:  # Only concatenate if records is not empty
            # Filter out empty dataframes from records before concatenation
            non_empty_records = [r for r in records if not r.empty]
            if non_empty_records:
                df_list.append(pd.concat(non_empty_records, axis=1, sort=False))

    if len(df_list) == 0:
        return pd.DataFrame()

    # Filter out empty DataFrames before concatenation
    non_empty_df_list = [df for df in df_list if not df.empty]
    if len(non_empty_df_list) == 0:
        return pd.DataFrame()

    # To avoid FutureWarning about concatenating with empty or all-NA entries,
    # we need to handle the case where DataFrames might have different dtypes
    # or contain all-NA columns
    if len(non_empty_df_list) == 1:
        return non_empty_df_list[0]

    # For multiple DataFrames, ensure consistent data types across columns
    # This is the recommended approach to avoid the FutureWarning
    # First, identify all columns across all DataFrames
    all_columns = set()
    for df in non_empty_df_list:
        all_columns.update(df.columns)

    # Reindex each DataFrame to have all columns, filling missing with NaN
    aligned_dfs = []
    for df in non_empty_df_list:
        aligned_df = df.reindex(columns=sorted(all_columns))
        aligned_dfs.append(aligned_df)

    # Now concatenate the aligned DataFrames
    # To avoid the FutureWarning, we need to handle the case where some DataFrames
    # might have all-NA columns after alignment
    # Use copy=False to avoid the warning about empty or all-NA entries
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*DataFrame concatenation with empty or all-NA entries.*",
        )
        return pd.concat(aligned_dfs, ignore_index=False, sort=False)


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
    deci_var_df = instance.decision_variables_df
    kind_counts = deci_var_df.kind.value_counts()
    info = {
        "num_vars": len(deci_var_df),
        "num_binary": kind_counts.get(str(ommx_v1.DecisionVariable.BINARY), 0),
        "num_integer": kind_counts.get(str(ommx_v1.DecisionVariable.INTEGER), 0),
        "num_continuous": kind_counts.get(str(ommx_v1.DecisionVariable.CONTINUOUS), 0),
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
    }
