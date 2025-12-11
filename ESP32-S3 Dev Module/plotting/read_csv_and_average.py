
from typing import Dict, List, Tuple
import csv
import sys
import os


def read_csv_to_dicts(file_path: str, skip_lines: int = 14) -> Tuple[Dict[str, List], Dict[int, Dict[str, object]]]:
    """Read CSV and return data stored as column-wise dictionary and row-wise dictionary.

    Returns:
        columns: {'step': [...], 'time': [...], 'frequency': [...], 'capacity': [...]} with lists of values (floats for numeric where possible).
        rows: {0: {'step': ..., 'time': ..., 'frequency': ..., 'capacity': ...}, ...}
    """
    r"""read_csv_and_average.py

    Reads a CSV file where the first 14 lines are text (to be skipped), then the data rows have 4 columns:
        step, time, frequency, capacity

    It stores the data in dictionaries (column-wise and row-wise) and computes the average frequency and capacity.

    Usage (PowerShell / Windows):
        & "./.venv/Scripts/python.exe" "plotting/read_csv_and_average.py" "log/100pF.csv" --unit pF

    Options:
        --unit {F,pF}   Unit for printed capacitance (default: pF)
        --skip-lines N  Number of header lines to skip (default: 14)
        file             Path to CSV file (optional; default: log/100pF.csv relative to repo root)
    """
    from typing import Dict, List, Tuple, Optional
    import csv
    import sys
    import os
    import argparse


    def read_csv_to_dicts(file_path: str, skip_lines: int = 14) -> Tuple[Dict[str, List], Dict[int, Dict[str, object]]]:
        """Read CSV and return data stored as column-wise dictionary and row-wise dictionary.

        Returns:
            columns: {'step': [...], 'time': [...], 'frequency': [...], 'capacity': [...]} with lists of values (floats for numeric where possible).
            rows: {0: {'step': ..., 'time': ..., 'frequency': ..., 'capacity': ...}, ...}
        """
        columns = {'step': [], 'time': [], 'frequency': [], 'capacity': []}
        rows = {}

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
r"""read_csv_and_average.py

Reads a CSV file where the first 14 lines are text (to be skipped), then the data rows have 4 columns:
    step, time, frequency, capacity

It stores the data in dictionaries (column-wise and row-wise) and computes the average frequency and capacitance.

Usage (PowerShell / Windows):
    & "./.venv/Scripts/python.exe" "plotting/read_csv_and_average.py" "log/100pF.csv" --unit pF

Options:
    --unit {F,pF}   Unit for printed capacitance (default: pF)
    --skip-lines N  Number of header lines to skip (default: 14)
    file             Path to CSV file (optional; default: log/100pF.csv relative to repo root)
"""
from typing import Dict, List, Tuple, Optional
import csv
import sys
import os
import argparse


def read_csv_to_dicts(file_path: str, skip_lines: int = 14) -> Tuple[Dict[str, List], Dict[int, Dict[str, object]]]:
    """Read CSV and return data stored as column-wise dictionary and row-wise dictionary.

    Returns:
        columns: {'step': [...], 'time': [...], 'frequency': [...], 'capacity': [...]} with lists of values (floats for numeric where possible).
        rows: {0: {'step': ..., 'time': ..., 'frequency': ..., 'capacity': ...}, ...}
    """
    columns = {'step': [], 'time': [], 'frequency': [], 'capacity': []}
    rows = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # skip header lines
        for _ in range(skip_lines):
            try:
                next(reader)
            except StopIteration:
                # file had less than skip_lines lines
                return columns, rows

        row_index = 0
        for raw_row in reader:
            # skip empty rows
            if not raw_row or all(not cell.strip() for cell in raw_row):
                continue

            # Expect at least 4 columns; ignore extras
            if len(raw_row) < 4:
                # try to be permissive: if the row is a single string with separators, try splitting
                joined = raw_row[0]
                parts = [p.strip() for p in joined.replace(';', ',').split(',') if p.strip()]
                if len(parts) < 4:
                    # malformed row: skip
                    continue
                else:
                    row = parts[:4]
            else:
                row = [cell.strip() for cell in raw_row[:4]]

            # parse values
            step_val = row[0]
            time_val = row[1]

            # frequency and capacity should be numeric; try to convert
            def to_float(x: Optional[str]):
                if x is None or x == '':
                    return None
                # handle commas as decimal separators
                x2 = x.replace(',', '.')
                try:
                    return float(x2)
                except ValueError:
                    return None

            freq_val = to_float(row[2])
            cap_val = to_float(row[3])

            columns['step'].append(step_val)
            columns['time'].append(time_val)
            columns['frequency'].append(freq_val)
            columns['capacity'].append(cap_val)

            rows[row_index] = {
                'step': step_val,
                'time': time_val,
                'frequency': freq_val,
                'capacity': cap_val,
            }
            row_index += 1

    return columns, rows


def safe_average(values: List[float]) -> float:
    """Compute average ignoring None values. Returns float('nan') if no valid numbers."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return float('nan')
    return sum(nums) / len(nums)


def format_capacitance(cap_F: float, unit: str) -> Tuple[float, str]:
    """Return capacitance value converted to requested unit and a label.

    Supported units: 'F' (farads), 'pF' (picofarads)
    """
    if unit == 'F':
        return cap_F, 'F'
    elif unit == 'pF':
        return cap_F * 1e12, 'pF'
    else:
        # fallback: return farads
        return cap_F, 'F'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read CSV and compute average frequency and capacitance.')
    parser.add_argument('file', nargs='?', default=None, help='Path to CSV file (optional).')
    parser.add_argument('--skip-lines', '-s', type=int, default=14, help='Number of header lines to skip (default: 14)')
    parser.add_argument('--unit', '-u', choices=['F', 'pF'], default='pF', help='Unit for printed capacitance (default: pF)')
    args = parser.parse_args()

    # default file path relative to repo root
    default_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', '68pF.csv')

    path = args.file if args.file is not None else default_csv

    try:
        cols, rows = read_csv_to_dicts(path, skip_lines=args.skip_lines)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)

    avg_freq = safe_average(cols['frequency'])
    avg_cap = safe_average(cols['capacity'])  # in farads as in logs

    cap_value, cap_label = format_capacitance(avg_cap, args.unit)

    print(f"Read {len(rows)} data rows from: {path}")
    print(f"Average frequency: {avg_freq:.6g} Hz")
    # print chosen unit and also show F for reference
    print(f"Average capacitance: {cap_value:.6g} {cap_label}  (raw: {avg_cap:.6g} F)")

    # Optionally, for programmatic use, one can import this module and call read_csv_to_dicts()
