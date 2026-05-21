import json
import os
import tempfile
from pathlib import Path


def _iter_split_records(split_json_path):
    split_json_path = Path(split_json_path)
    if not split_json_path.is_file():
        raise FileNotFoundError(f"Split json file not found: {split_json_path}")

    first_char = None
    with split_json_path.open("r", encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break

    if first_char == "[":
        with split_json_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {split_json_path}")
        for record in records:
            yield record
        return

    with split_json_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {split_json_path} line {line_no}: {exc}") from exc


def extract_sequence_ids(split_json_path):
    sequence_ids = set()
    for record in _iter_split_records(split_json_path):
        if isinstance(record, dict):
            raw_file = record.get("raw_file")
        else:
            raw_file = record

        if not isinstance(raw_file, str):
            raise ValueError(f"Split record does not contain a valid raw_file: {record}")

        parts = raw_file.strip("/").split("/")
        if len(parts) < 3:
            raise ValueError(f"Unable to extract sequence id from raw_file={raw_file}")
        sequence_ids.add(parts[0])

    if not sequence_ids:
        raise RuntimeError(f"No sequence ids were found in split json: {split_json_path}")

    return sequence_ids


def build_sequence_split_index(base_index_file, split_json_path, output_dir, split_name=None):
    base_index_file = Path(base_index_file)
    if not base_index_file.is_file():
        raise FileNotFoundError(f"Base index file not found: {base_index_file}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_ids = extract_sequence_ids(split_json_path)
    split_stem = split_name or Path(split_json_path).stem
    output_path = output_dir / f"{base_index_file.stem}__{split_stem}.txt"

    filtered_lines = []
    with base_index_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            rel_path = line.strip()
            if not rel_path:
                continue
            sequence_id = rel_path.lstrip("/").split("/", 1)[0]
            if sequence_id in sequence_ids:
                filtered_lines.append(rel_path)

    if not filtered_lines:
        raise RuntimeError(
            f"No samples from {base_index_file} matched sequences extracted from {split_json_path}"
        )

    payload = "\n".join(filtered_lines) + "\n"
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            if handle.read() == payload:
                return str(output_path)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_dir,
            prefix=output_path.stem + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, output_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return str(output_path)
