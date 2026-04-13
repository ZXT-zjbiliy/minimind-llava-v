import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--head", type=int, default=2)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(f"pandas is required: {exc}") from exc

    df = pd.read_parquet(args.path)
    print(f"path: {args.path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    records = df.head(args.head).to_dict(orient="records")
    for idx, record in enumerate(records):
        print(f"sample[{idx}]")
        for key, value in record.items():
            if isinstance(value, str):
                preview = value[:300].replace("\n", "\\n")
            else:
                preview = repr(value)[:300]
            print(json.dumps({
                "field": key,
                "type": type(value).__name__,
                "preview": preview,
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
