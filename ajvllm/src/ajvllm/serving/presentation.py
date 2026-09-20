"""Human-readable display fields alongside machine-readable measurements."""


def memory_display(value):
    if isinstance(value, dict):
        result = {key: memory_display(item) for key, item in value.items()}
        for key, item in value.items():
            if key.endswith("_bytes") and isinstance(item, (int, float)):
                unit, divisor = ("GiB", 1024**3) if abs(item) >= 1024**3 else ("MiB", 1024**2)
                result[key.removesuffix("_bytes") + "_display"] = f"{item / divisor:.2f} {unit}"
        return result
    return value


def request_timing(output):
    first, finish = output.first_token_time, output.finish_time
    durations = {
        "ttft_s": None if first is None else first - output.arrival_time,
        "latency_s": None if finish is None else finish - output.arrival_time,
        "decode_s": None if first is None or finish is None else finish - first,
    }
    count = len(output.output_token_ids)
    durations["tpot_s"] = durations["decode_s"] / (count - 1) if count > 1 and finish is not None else None
    return durations | {"display": {key: f"{value:.6f} s" for key, value in durations.items() if value is not None}}
