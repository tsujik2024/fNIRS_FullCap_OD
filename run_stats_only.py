import os
import pandas as pd
from fnirs_FullCap_2025.processing.statistics import StatisticsCalculator

# === Set your path ===
output_base_dir = "/Users/tsujik/Documents/auttest"
stats_collector = StatisticsCalculator()

# === Find all _processed.csv files ===
processed_csv_files = []
for root, dirs, files in os.walk(output_base_dir):
    for file in files:
        if file.endswith("_processed.csv") and "bad_SCI" not in file:
            full_path = os.path.join(root, file)
            processed_csv_files.append(full_path)
            print(f"Found: {full_path}")

print(f"🔍 Found {len(processed_csv_files)} processed CSVs")

if not processed_csv_files:
    print("❌ No processed CSV files found!")
    exit()

# === Load and combine all CSV files ===
print("📊 Loading and combining CSV files...")
all_dataframes = []

for csv_file in processed_csv_files:
    try:
        df = pd.read_csv(csv_file)

        # Metadata columns
        if 'subject_id' not in df.columns:
            subject_id = os.path.basename(csv_file).split('_')[0] + '_' + os.path.basename(csv_file).split('_')[1]
            df['subject_id'] = subject_id

        if 'visit' not in df.columns:
            visit = next((part for part in csv_file.split('/') if part.startswith('Visit')), 'Unknown')
            df['visit'] = visit

            filename = os.path.basename(csv_file)
            condition = 'Unknown'

            # Prioritize Cue_Walking before Walking
            if 'Cue_Walking' in filename:
                if 'DT1' in filename:
                    condition = 'Cue_Walking_DT1'
                elif 'DT3' in filename:
                    condition = 'Cue_Walking_DT3'
                elif 'ST' in filename:
                    condition = 'Cue_Walking_ST'
                else:
                    condition = 'Cue_Walking'
            elif 'Walking_ST' in filename or 'WalkingST' in filename:
                condition = 'Walking_ST'
            elif 'Walking_DT1' in filename or 'WalkingDT1' in filename:
                condition = 'Walking_DT1'
            elif 'Walking_DT2' in filename or 'WalkingDT2' in filename:
                condition = 'Walking_DT2'
            elif 'Walking_DT3' in filename or 'WalkingDT3' in filename:
                condition = 'Walking_DT3'
            elif 'Sitting' in filename:
                condition = 'Sitting'
            elif 'Standing' in filename:
                condition = 'Standing'
            else:
                parts = filename.replace('_processed.csv', '').split('_')
                for i in range(len(parts) - 1):
                    if 'Walking' in parts[i] or 'Cue' in parts[i]:
                        condition = f"{parts[i]}_{parts[i + 1]}"
                        break

            df['condition'] = condition
            df['Condition'] = condition

        if 'file_path' not in df.columns:
            df['file_path'] = csv_file

        all_dataframes.append(df)

    except Exception as e:
        print(f"⚠️  Error loading {csv_file}: {e}")

if not all_dataframes:
    print(" No CSV files could be loaded!")
    exit()

# === Combine all dataframes ===
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"📈 Combined dataset shape: {combined_df.shape}")

# === Rename columns for compatibility ===
combined_df = combined_df.rename(columns={
    'grand_oxy': 'grand oxy',
    'grand_deoxy': 'grand deoxy'
})
# === Grouped stats: grand oxy + all regional oxy channels ===
print("🧮 Calculating grouped statistics (grand + regional HbO)...")
grouped_stats = []
group_cols = ['subject_id', 'visit', 'condition']

# Define all HbO columns to analyze
hbo_columns = [col for col in combined_df.columns if col.endswith('_oxy') or col == 'grand oxy']

for keys, group in combined_df.groupby(group_cols):
    subject_id, visit, condition = keys
    total_samples = len(group)
    half = total_samples // 2

    row = {
        'subject_id': subject_id,
        'visit': visit,
        'condition': condition
    }

    for col in hbo_columns:
        try:
            row[f'{col} - Overall Mean'] = group[col].mean()
            row[f'{col} - First Half Mean'] = group[col].iloc[:half].mean()
            row[f'{col} - Second Half Mean'] = group[col].iloc[half:].mean()
        except Exception as e:
            print(f"⚠️ Skipping {col} due to error: {e}")

    grouped_stats.append(row)

stats_df = pd.DataFrame(grouped_stats)
stats_df['Condition'] = stats_df['condition']


# === Save to CSV ===
output_file = os.path.join(output_base_dir, "all_subjects_statistics.csv")
stats_df.to_csv(output_file, index=False)
print(f"💾 Saved statistics to: {output_file}")

# === Summary of structure ===
hbo_channels = [col for col in combined_df.columns if col.startswith('CH') and 'HbO' in col]
hhb_channels = [col for col in combined_df.columns if col.startswith('CH') and 'HHb' in col]
left_regions = [col for col in combined_df.columns if '_L_' in col]
right_regions = [col for col in combined_df.columns if '_R_' in col]
combined_regions = [col for col in combined_df.columns if '_combined_' in col]
grand_measures = [col for col in combined_df.columns if col.startswith('grand_')]

summary_info = {
    'total_subjects': len(combined_df['subject_id'].unique()),
    'total_visits': len(combined_df['visit'].unique()),
    'total_rows': len(combined_df),
    'individual_hbo_channels': len(hbo_channels),
    'individual_hhb_channels': len(hhb_channels),
    'left_regions': len(left_regions),
    'right_regions': len(right_regions),
    'combined_regions': len(combined_regions),
    'grand_measures': len(grand_measures),
    'brain_regions': sorted(set(col.split('_')[0] for col in left_regions + right_regions))
}

summary_df = pd.DataFrame([summary_info])
summary_file = os.path.join(output_base_dir, "analysis_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f" Saved analysis summary to: {summary_file}")

# === Generate summary sheets ===
try:
    stats_collector.create_summary_sheets(stats_df, output_base_dir)
    print(" Summary sheets created successfully!")
except Exception as e:
    print(f"  Warning: Could not create summary sheets: {e}")

print(" Stats-only summary complete.")
