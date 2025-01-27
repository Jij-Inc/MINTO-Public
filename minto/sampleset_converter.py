import jijmodeling as jm
import ommx.v1
import ommx.v1.sample_set_pb2 as sample_set_pb2
from ommx.v1.constraint_pb2 import Equality


def convert_sampleset_jijmodeling_to_ommx(
    jm_sampleset: jm.SampleSet | jm.experimental.SampleSet,
) -> ommx.v1.SampleSet:
    if isinstance(jm_sampleset, jm.SampleSet):
        jm_sampleset = jm.experimental.from_old_sampleset(jm_sampleset)

    samples_value = _extract_deci_var_from_sampleset(jm_sampleset)

    objectives: list[sample_set_pb2.SampledValues.SampledValuesEntry] = []
    for sample_id, sample in enumerate(jm_sampleset):
        objectives.append(
            sample_set_pb2.SampledValues.SampledValuesEntry(
                value=sample.eval.objective, ids=[sample_id]
            )
        )

    constraints, feasible = _extract_constraints(jm_sampleset)

    return ommx.v1.SampleSet(
        raw=sample_set_pb2.SampleSet(
            objectives=sample_set_pb2.SampledValues(entries=objectives),
            decision_variables=samples_value,
            constraints=constraints,
            feasible=feasible,
            feasible_unrelaxed=feasible,
        )
    )


def _extract_var_map(
    deci_var_value: jm.experimental.SparseVarValues,
    var_map: dict[str, dict[tuple[int, ...], int]],
    deci_var_map: dict,
) -> tuple[
    dict[str, dict[tuple[int, ...], int]],
    dict[int, ommx.v1.DecisionVariable],
]:
    var_name = deci_var_value.name

    # create var_map and deci_var
    var_map[var_name] = var_map.get(var_name, {})
    var_id = sum(len(v) for v in var_map.values())
    var_id_counter = len(var_map[var_name])
    for subscripts, value in deci_var_value.values.items():
        var_map[var_name][subscripts] = var_map[var_name].get(subscripts, var_id)
        ox_deci_var: ommx.v1.DecisionVariable
        if deci_var_value.var_type == jm.experimental.VarType.BINARY:
            ox_deci_var = ommx.v1.DecisionVariable.binary(
                var_id,
                name=var_name,
                subscripts=list(subscripts),
            )
        elif deci_var_value.var_type == jm.experimental.VarType.INTEGER:
            ox_deci_var = ommx.v1.DecisionVariable.integer(
                var_id,
                name=var_name,
                subscripts=list(subscripts),
            )
        elif deci_var_value.var_type == jm.experimental.VarType.CONTINUOUS:
            ox_deci_var = ommx.v1.DecisionVariable.continuous(
                var_id,
                name=var_name,
                subscripts=list(subscripts),
            )
        else:
            raise ValueError(f"Unsupported var_type: {deci_var_value.var_type}")
        deci_var_map[var_map[var_name][subscripts]] = deci_var_map.get(
            var_map[var_name][subscripts], ox_deci_var
        )
        if var_id_counter < len(var_map[var_name]):
            var_id += 1
            var_id_counter = len(var_map[var_name])
    return var_map, deci_var_map


def _extract_deci_var_from_sampleset(sampleset: jm.experimental.SampleSet):

    var_id = 0
    var_map: dict = {}
    deci_var_map: dict = {}
    nonzero_map: dict[int, list[int]] = {}
    samples: dict[int, list[sample_set_pb2.SampledValues.SampledValuesEntry]] = {}
    for sample_id, sample in enumerate(sampleset):
        for var_name, values in sample.var_values.items():
            var_map, deci_var_map = _extract_var_map(values, var_map, deci_var_map)

            for subscripts, value in values.values.items():
                var_id = var_map[var_name][subscripts]
                entry = sample_set_pb2.SampledValues.SampledValuesEntry(
                    value=value, ids=[sample_id]
                )
                if var_id not in samples:
                    samples[var_id] = []
                samples[var_id].append(entry)

                if var_id not in nonzero_map:
                    nonzero_map[var_id] = []
                nonzero_map[var_id].append(sample_id)

    for var_id, sample_ids in nonzero_map.items():
        samples[var_id].append(
            sample_set_pb2.SampledValues.SampledValuesEntry(
                value=0,
                ids=[
                    sample_id
                    for sample_id in range(len(sampleset))
                    if sample_id not in sample_ids
                ],
            )
        )
    return [
        sample_set_pb2.SampledDecisionVariable(
            decision_variable=deci_var_map[var_id].raw,
            samples=sample_set_pb2.SampledValues(entries=values),
        )
        for var_id, values in samples.items()
    ]


def _extract_constraints(sampleset: jm.experimental.SampleSet):
    violations: dict[int, tuple[dict, list, dict]] = {}
    feasibles: dict[int, bool] = {}
    for sample_id, sample in enumerate(sampleset):
        for const_id, (const_name, violation) in enumerate(
            sample.eval.constraints.items()
        ):
            for subscripts, value in violation.expr_values.items():
                if const_id not in violations:
                    meta_data = {
                        "equality": Equality.EQUALITY_UNSPECIFIED,
                        "name": const_name,
                        "subscripts": list(subscripts),
                        "used_decision_variable_ids": [],
                        "removed_reason_parameters": {},
                        "parameters": {},
                    }
                    violations[const_id] = (meta_data, [], {})

                violations[const_id][1].append(value)
                violations[const_id][2][sample_id] = abs(value) <= 1e-8

        feasibles[sample_id] = sample.is_feasible()

    # covert to ommx.v1.sample_set_pb2.SampledConstraint
    ox_constraints: list = []
    for const_id, (meta_data, values, _feasibles) in violations.items():
        ox_constraints.append(
            sample_set_pb2.SampledConstraint(
                id=const_id,
                evaluated_values=sample_set_pb2.SampledValues(
                    entries=[
                        sample_set_pb2.SampledValues.SampledValuesEntry(
                            value=value,
                            ids=[sample_id],
                        )
                        for sample_id, value in enumerate(values)
                    ],
                ),
                feasible=_feasibles,
                **meta_data,
            )
        )

    return ox_constraints, feasibles
