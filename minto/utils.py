import typing as typ


def type_check(ob_type_list: list[tuple[str, typ.Any, typ.Type]]):
    for name, obj, type_ in ob_type_list:
        if not isinstance(obj, type_):
            raise TypeError(f"{name} should be {type_} type, but got {type(obj)}.")
