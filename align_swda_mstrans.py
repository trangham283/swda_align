import glob
import os
import pandas as pd
import re
from difflib import SequenceMatcher


# from https://github.com/cgpotts/swda/blob/master/swda.py
def damsl_act_tag(act_tag):
    """
    Seeks to duplicate the tag simplification described at the
    Coders' Manual: http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
    """
    d_tags = []
    tags = re.split(r"\s*[,;]\s*", act_tag)
    for tag in tags:
        if tag in ("qy^d", "qw^d", "b^m"):
            pass
        elif tag == "nn^e":
            tag = "ng"
        elif tag == "ny^e":
            tag = "na"
        else:
            tag = re.sub(r"(.)\^.*", r"\1", tag)
            tag = re.sub(r"[\(\)@*]", "", tag)
            if tag in ("qr", "qy"):
                tag = "qy"
            elif tag in ("fe", "ba"):
                tag = "ba"
            elif tag in ("oo", "co", "cc"):
                tag = "oo_co_cc"
            elif tag in ("fx", "sv"):
                tag = "sv"
            elif tag in ("aap", "am"):
                tag = "aap_am"
            elif tag in ("arp", "nd"):
                tag = "arp_nd"
            elif tag in ("fo", "o", "fw", '"', "by", "bc"):
                tag = 'fo_o_fw_"_by_bc'
        d_tags.append(tag)
    # Dan J says (p.c.) that it makes sense to take the first;
    # there are only a handful of examples with 2 tags here.
    return d_tags[0]


# normalize SWDA transcripts
def clean_text(text):
    # remove disfluency markers
    # see https://github.com/cgpotts/swda/blob/master/swda.py#L351
    text = re.sub(r"([+/\}\[\]]|\{\w)", "", text)
    # remove punctuation except -', convert to lowercase
    text = re.sub(r"[^\w\s'-]", "", text.lower())
    # replace multiple whitespace with single space and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


# unroll utterances into words for alignment
def unroll_words(df_side):
    list_row = []
    for i, row in df_side.iterrows():
        words = row.text_norm.split()
        for w in words:
            list_row.append(
                {
                    "utterance_index": row.utterance_index,
                    "subutterance_index": row.subutterance_index,
                    "caller": row.caller,
                    "act_tag": row.act_tag,
                    "word": w,
                }
            )
    return pd.DataFrame(list_row)


# sequence alignment function for word-level alignment
# claude-generated
def align_sequences(seq1, seq2):
    """
    Align two sequences using difflib.SequenceMatcher
    seq1: list of words from SWDA
    seq2: list of words from transcript
    Returns: list of tuples (word1, word2, alignment_type)
    """
    matcher = SequenceMatcher(None, seq1, seq2)
    alignment = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Match
            for k in range(i2 - i1):
                alignment.append((seq1[i1 + k], seq2[j1 + k], "match"))
        elif tag == "replace":
            # Mismatch - align 1:1 where possible
            len1, len2 = i2 - i1, j2 - j1
            for k in range(max(len1, len2)):
                word1 = seq1[i1 + k] if k < len1 else None
                word2 = seq2[j1 + k] if k < len2 else None
                if word1 and word2:
                    alignment.append((word1, word2, "mismatch"))
                elif word1:
                    alignment.append((word1, None, "deletion"))
                else:
                    alignment.append((None, word2, "insertion"))
        elif tag == "delete":
            # Deletion
            for k in range(i1, i2):
                alignment.append((seq1[k], None, "deletion"))
        elif tag == "insert":
            # Insertion
            for k in range(j1, j2):
                alignment.append((None, seq2[k], "insertion"))

    return alignment


# create aligned dataframe
# claude-generated
def create_aligned_dataframe(swda_words, trans_df, speaker):
    """
    Create aligned dataframe combining SWDA annotations with transcript timing
    """
    swda_sequence = swda_words["word"].tolist()
    trans_sequence = trans_df["transcription"].tolist()

    alignment = align_sequences(swda_sequence, trans_sequence)

    aligned_data = []
    swda_idx = 0
    trans_idx = 0

    for swda_word, trans_word, align_type in alignment:
        row = {
            "speaker": speaker,
            "alignment_type": align_type,
            "swda_word": swda_word,
            "trans_word": trans_word,
        }

        # Add SWDA information if available
        if swda_word is not None and swda_idx < len(swda_words):
            swda_row = swda_words.iloc[swda_idx]
            row.update(
                {
                    "utterance_index": swda_row["utterance_index"],
                    "subutterance_index": swda_row["subutterance_index"],
                    "act_tag": swda_row["act_tag"],
                }
            )
            swda_idx += 1
        else:
            row.update(
                {
                    "utterance_index": None,
                    "subutterance_index": None,
                    "act_tag": None,
                }
            )

        # Add transcript timing information if available
        if trans_word is not None and trans_idx < len(trans_df):
            trans_row = trans_df.iloc[trans_idx]
            row.update(
                {
                    "conv_no": trans_row["conv_no"],
                    "turn_id": trans_row["turn_id"],
                    "start": trans_row["start"],
                    "end": trans_row["end"],
                }
            )
            trans_idx += 1
        else:
            row.update(
                {
                    "conv_no": None,
                    "turn_id": None,
                    "start": None,
                    "end": None,
                }
            )

        aligned_data.append(row)

    return pd.DataFrame(aligned_data)


# helper function to parse transcript files
def read_transcript_file(file_path, conv_no, speaker):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            turn_id = parts[0]
            start = float(parts[1])
            end = float(parts[2])
            transcript = " ".join(parts[3:])  # rest is transcript
            # skip turns without speech
            if transcript == "[silence]" or transcript == "[noise]":
                continue
            data.append(
                {
                    "conv_no": conv_no,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "transcription": transcript,
                }
            )
    df = pd.DataFrame(data)
    return df


def load_swda_mapping():
    """Load SWDA CSV files and create conversation number mapping"""
    swda_csvs = glob.glob("swda/swda/sw*/*.csv")
    # csv filenames are in the form 'swda/swda/sw06utt/sw_0682_4660.utt.csv'
    swda_convs = [os.path.basename(x).split("_")[-1].split(".")[0] for x in swda_csvs]
    # 1155 conversations with DA tags
    return dict(zip(swda_convs, swda_csvs))


def get_transcript_conversations(trans_dir="swb_ms98_transcriptions"):
    """Get list of available transcript conversations"""
    trans_convs = []
    sub_dirs = os.listdir(trans_dir)

    for sub_dir in sub_dirs:
        if not sub_dir.isdigit():
            continue
        conv_dirs = os.listdir(os.path.join(trans_dir, sub_dir))
        trans_convs.extend(conv_dirs)

    return trans_convs


def process_conversation(conv_num, conv2csv, trans_dir="swb_ms98_transcriptions"):
    """Process a single conversation and return aligned dataframe"""
    # Find subdirectory for this conversation
    sub_dir = None
    for sd in os.listdir(trans_dir):
        if sd.isdigit() and conv_num in os.listdir(os.path.join(trans_dir, sd)):
            sub_dir = sd
            break

    if sub_dir is None:
        print(f"Warning: Could not find subdirectory for conversation {conv_num}")
        return None

    # Read transcript files
    sideA_file = os.path.join(
        trans_dir, sub_dir, conv_num, f"sw{conv_num}A-ms98-a-word.text"
    )
    sideB_file = os.path.join(
        trans_dir, sub_dir, conv_num, f"sw{conv_num}B-ms98-a-word.text"
    )

    try:
        dfA = read_transcript_file(sideA_file, conv_num, "A")
        dfB = read_transcript_file(sideB_file, conv_num, "B")
    except FileNotFoundError as e:
        print(
            f"Warning: Could not find transcript files for conversation {conv_num}: {e}"
        )
        return None

    # Read SWDA annotations
    try:
        swda_df = pd.read_csv(conv2csv[conv_num])
    except FileNotFoundError:
        print(f"Warning: Could not find SWDA file for conversation {conv_num}")
        return None

    # Process SWDA data
    swda_df["text_norm"] = swda_df["text"].apply(clean_text)
    swda_A = swda_df[swda_df.caller == "A"]
    swda_B = swda_df[swda_df.caller == "B"]
    swda_words_A = unroll_words(swda_A)
    swda_words_B = unroll_words(swda_B)

    # Create aligned dataframes
    aligned_A = create_aligned_dataframe(swda_words_A, dfA, "A")
    aligned_B = create_aligned_dataframe(swda_words_B, dfB, "B")

    # Combine both speakers
    aligned_conv = (
        pd.concat([aligned_A, aligned_B])
        .sort_values(["start", "speaker"])
        .reset_index(drop=True)
    )

    return aligned_conv


def main():
    """Main processing function"""
    print("Loading SWDA conversation mapping...")
    conv2csv = load_swda_mapping()
    print(f"Found {len(conv2csv)} SWDA conversations")

    print("Getting transcript conversations...")
    trans_convs = get_transcript_conversations()
    print(f"Found {len(trans_convs)} transcript conversations")

    # Process conversations that have both SWDA and transcript data
    processed_count = 0
    for conv_num in trans_convs:
        if conv_num not in conv2csv:
            continue

        aligned_conv = process_conversation(conv_num, conv2csv)
        if aligned_conv is not None:
            processed_count += 1
            print(
                f"Processed conversation {conv_num}: {len(aligned_conv)} aligned words"
            )
            # Uncomment to save individual conversation files
            # aligned_conv.to_csv(f'aligned_conv_{conv_num}.csv', index=False)

    print(f"Successfully processed {processed_count} conversations")


if __name__ == "__main__":
    main()
