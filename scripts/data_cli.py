import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
CLI for data loading and cleaning
"""
import argparse
from src import data

def main():
    parser = argparse.ArgumentParser(description="Data loading and cleaning utility.")
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--output', help='Output cleaned CSV file path', default=None)
    args = parser.parse_args()
    df = data.load_and_clean(args.input)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Cleaned data saved to {args.output}")
    else:
        print(df.head())

if __name__ == "__main__":
    main()
