import pandas as pd


def to_frame(payload):
    dimensions = payload["id"]
    sizes = payload["size"]
    codes = []
    for name in dimensions:
        index = payload["dimension"][name]["category"]["index"]
        if isinstance(index, list):
            ordered = list(index)
        else:
            ordered = [code for code, _ in sorted(index.items(), key=lambda item: item[1])]
        codes.append(ordered)
    values = payload.get("value", {})
    if isinstance(values, list):
        lookup = {position: value for position, value in enumerate(values) if value is not None}
    else:
        lookup = {int(position): value for position, value in values.items() if value is not None}
    strides = []
    stride = 1
    for size in reversed(sizes):
        strides.append(stride)
        stride *= size
    strides = list(reversed(strides))
    rows = []
    for position, value in lookup.items():
        remainder = position
        row = {}
        for name, step, options in zip(dimensions, strides, codes):
            row[name] = options[remainder // step]
            remainder %= step
        row["value"] = float(value)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*dimensions, "value"])
    return pd.DataFrame(rows)

