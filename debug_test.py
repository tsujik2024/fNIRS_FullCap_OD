#!/usr/bin/env python3
"""
Debug script to trace file flow in your pipeline
"""
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def debug_pipeline_files(input_dir, output_dir):
    """Debug what files exist where"""
    print("=" * 80)
    print("DEBUGGING PIPELINE FILE FLOW")
    print("=" * 80)

    # 1. Check input directory
    print(f"\n1. INPUT DIRECTORY: {input_dir}")
    if os.path.exists(input_dir):
        txt_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        print(f"   Found {len(txt_files)} .txt files")
        for txt_file in txt_files[:3]:
            print(f"   - {txt_file}")
        if len(txt_files) > 3:
            print(f"   - ... and {len(txt_files) - 3} more")
    else:
        print(f"   ERROR: Input directory doesn't exist!")
        return

    # 2. Check output directory
    print(f"\n2. OUTPUT DIRECTORY: {output_dir}")
    if os.path.exists(output_dir):
        csv_files = []
        pdf_files = []
        other_files = []

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith('.csv'):
                    csv_files.append(full_path)
                elif file.endswith('.pdf'):
                    pdf_files.append(full_path)
                else:
                    other_files.append(full_path)

        print(f"   Found {len(csv_files)} .csv files")
        print(f"   Found {len(pdf_files)} .pdf files")
        print(f"   Found {len(other_files)} other files")

        # Show CSV files specifically
        processed_csv_files = [f for f in csv_files if '_processed.csv' in f]
        print(f"   Found {len(processed_csv_files)} _processed.csv files:")
        for csv_file in processed_csv_files[:5]:
            print(f"   - {os.path.basename(csv_file)}")

        if len(processed_csv_files) > 5:
            print(f"   - ... and {len(processed_csv_files) - 5} more")

    else:
        print(f"   Output directory doesn't exist yet")

    print("=" * 80)
    return txt_files if os.path.exists(input_dir) else []


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python debug_test.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    debug_pipeline_files(input_dir, output_dir)