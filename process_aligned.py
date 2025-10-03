import glob
import os
import pandas as pd


def fill_act_tag_by_turn(df):
    """
    Fill in act_tag columns by looking both forward and backward
    for rows with the same turn_id.
    """
    df = df.copy()

    # Forward pass: fill from following row
    for i in range(len(df) - 1, -1, -1):
        if (
            i < len(df) - 1
            and pd.isna(df.loc[i, "act_tag"])
            and df.loc[i, "turn_id"] == df.loc[i + 1, "turn_id"]
        ):
            df.loc[i, "act_tag"] = df.loc[i + 1, "act_tag"]
            df.loc[i, "utterance_index"] = df.loc[i + 1, "utterance_index"]
            df.loc[i, "subutterance_index"] = df.loc[i + 1, "subutterance_index"]

    # Backward pass: fill from preceding row
    for i in range(1, len(df)):
        if (
            pd.isna(df.loc[i, "act_tag"])
            and df.loc[i, "turn_id"] == df.loc[i - 1, "turn_id"]
        ):
            df.loc[i, "act_tag"] = df.loc[i - 1, "act_tag"]
            df.loc[i, "utterance_index"] = df.loc[i - 1, "utterance_index"]
            df.loc[i, "subutterance_index"] = df.loc[i - 1, "subutterance_index"]

    # Third pass: find nearest row with same turn_id for any remaining NaNs
    # We don't do this before the first two passes because it is possible
    # for turn_id to map to different (utterance_index, subutterance_index)s,
    # which we prioritize matching because act_tags follow those indices
    for i in range(len(df)):
        if pd.isna(df.loc[i, "act_tag"]):
            current_turn_id = df.loc[i, "turn_id"]
            # Search for nearest row with same turn_id
            min_distance = float("inf")
            nearest_idx = None

            for j in range(len(df)):
                if (
                    j != i
                    and df.loc[j, "turn_id"] == current_turn_id
                    and pd.notna(df.loc[j, "act_tag"])
                ):
                    distance = abs(j - i)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_idx = j

            if nearest_idx is not None:
                df.loc[i, "act_tag"] = df.loc[nearest_idx, "act_tag"]
                df.loc[i, "utterance_index"] = df.loc[nearest_idx, "utterance_index"]
                df.loc[i, "subutterance_index"] = df.loc[
                    nearest_idx, "subutterance_index"
                ]

    return df


def group_by_utterance(df):
    """
    Group by utterance_index and subutterance_index, creating aggregated utterances.
    Returns a dataframe with columns: speaker, utterance_index, subutterance_index,
    start, end, turn_id, transcript_swda, transcript_ms, act_tag
    """
    grouped = df.groupby(["utterance_index", "subutterance_index"], dropna=False)

    rows = []
    for (utt_idx, sub_idx), group in grouped:
        row = {
            "speaker": group.iloc[0]["speaker"],
            "utterance_index": utt_idx,
            "subutterance_index": sub_idx,
            "start": group.iloc[0]["start"],
            "end": group.iloc[-1]["end"],
            "turn_id": group.iloc[0]["turn_id"],
            "transcript_swda": " ".join(group["swda_word"].dropna().astype(str)),
            "transcript_ms": " ".join(group["trans_word"].dropna().astype(str)),
            "act_tag": group.iloc[0]["act_tag"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Process all aligned conversation files and create turn-level aggregations."""
    # Create output directory
    os.makedirs("aligned_turns", exist_ok=True)

    # Load all aligned files
    aligned_files = sorted(glob.glob("aligned_words/aligned_conv_*.csv"))

    print(f"Processing {len(aligned_files)} files...")

    for file_path in aligned_files:
        # Extract conversation number from filename
        filename = os.path.basename(file_path)
        conv_num = filename.replace("aligned_conv_", "").replace(".csv", "")

        # Load dataframe
        df = pd.read_csv(file_path)

        # Drop rows where alignment_type == 'deletion'
        df = df[df["alignment_type"] != "deletion"]

        # Sort the dataframe by start
        df = df.sort_values("start").reset_index(drop=True)

        # Fill act_tag by turn_id
        df = fill_act_tag_by_turn(df)

        # Report and remove rows without act_tag
        missing_act_tag = df["act_tag"].isna().sum()
        if missing_act_tag > 0:
            print(
                f"  {conv_num}: unable to fill act_tag for {missing_act_tag}/{len(df)} words"
            )

        df = df[df["act_tag"].notna()]

        # Group by utterance
        df_utterances = group_by_utterance(df)

        # Write to new folder
        output_path = f"aligned_turns/aligned_turns_{conv_num}.csv"
        df_utterances.to_csv(output_path, index=False)

    print(f"\nProcessed {len(aligned_files)} conversations and saved to aligned_turns/")


if __name__ == "__main__":
    main()
