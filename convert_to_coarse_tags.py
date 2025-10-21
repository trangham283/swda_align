#!/usr/bin/env python3
"""
Convert act_tag to coarse labels:
- Fix continuation: when act_tag is "+", replace it with the previous turn's act_tag from the same speaker.
- Merge fine tags into coarse tags to 4 categories: question, answer, statement, other
"""

import pandas as pd
import glob
import os


def tag_mapping(tag):
    if tag.startswith("q"):
        return "question"
    elif tag.startswith("s"):
        return "statement"
    elif tag.startswith("a") or tag.startswith("n") or tag.startswith("b"):
        return "answer"
    else:
        return "other"


def process_csv_file(filepath, output_dir):
    """Process a single CSV file to fix continuation tags."""
    df = pd.read_csv(filepath)
    df["act_tag_merge"] = df["act_tag"]  # copy

    # Track the last act_tag for each speaker
    last_act_tag = {}

    for idx, row in df.iterrows():
        speaker = row["speaker"]
        act_tag = row["act_tag"]

        if act_tag == "+":
            # Replace with the last act_tag for this speaker
            if speaker in last_act_tag:
                df.at[idx, "act_tag_merge"] = last_act_tag[speaker]
            else:
                print(
                    f"Warning in {os.path.basename(filepath)}: Speaker {speaker} has '+' but no previous act_tag"
                )
        else:
            # Update the last act_tag for this speaker
            last_act_tag[speaker] = act_tag

    # Convert to coarse tags
    df["act_tag_merge"] = df["act_tag_merge"].apply(lambda x: tag_mapping(x))

    # Write file
    outpath = filepath.replace("aligned_turns", "coarse_tags")
    print(outpath)
    df.to_csv(outpath, index=False)
    return


def main():
    """Process all CSV files in the aligned_turns directory."""
    csv_files = glob.glob("aligned_turns/*.csv")

    output_dir = "coarse_tags"
    os.makedirs(output_dir, exist_ok=True)

    total_modified = 0
    for filepath in csv_files:
        process_csv_file(filepath, output_dir)


if __name__ == "__main__":
    main()
