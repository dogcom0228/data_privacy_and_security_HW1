import csv

def auto_cast(value):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return ""

    lower = v.lower()
    if lower in ("true", "false"):
        return lower == "true"

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        return value

def read_csv(file_path, mode='r', encoding='utf-8'):
    """Reads a CSV file and returns a list of dictionaries with auto type casting."""
    with open(file_path, mode=mode, encoding=encoding, newline='') as file:
        reader = csv.DictReader(file)
        return [
            {k: auto_cast(v) for k, v in row.items()}
            for row in reader
        ]

def format_value(val):
    if val is None:
        return ""
    if isinstance(val, tuple):
        return "-".join(str(v) for v in val)
    return str(val)
