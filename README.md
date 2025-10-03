# SWDA Alignment

Alignments for Switchboard Dialog Act (SWDA) tags with Mississippi State (MS-State) word-level transcripts with timing information.

## Overview

This project aligns the dialog act annotations from the [Switchboard Dialog Act Corpus](https://github.com/cgpotts/swda) with the word-level time-aligned transcripts from the [Mississippi State Switchboard annotations](https://isip.piconepress.com/projects/switchboard/). The result is a dataset where each utterance has:
- Dialog act tags (from SWDA)
- Word-level timing information (from MS-State transcripts)
- Speaker identification
- Aligned transcripts from both sources

## Data Sources

- **SWDA (Switchboard Dialog Act Corpus)**: Dialog act annotations for Switchboard conversations
  - 1,155 conversations with DA tags
  - Utterance-level annotations with act tags (statement, question, backchannel, etc.)

- **MS-State Transcripts**: Word-level time-aligned transcripts
  - 2,438 conversations available
  - Word-level start/end timestamps
  - Located in `swb_ms98_transcriptions/`

## Pipeline

### 1. Word-Level Alignment (`align_swda_mstrans.py`)

The main alignment script that:
- Loads SWDA annotations and MS-State transcripts
- Normalizes text for better matching (handles variants like "uh-huh", "um-hum", etc.)
- Performs sequence alignment using `difflib.SequenceMatcher`
- **Prioritizes longer consecutive match sequences** over isolated single-word matches
- Automatically detects and corrects speaker channel flips (84 conversations had flipped channels)
- Outputs word-level alignments to `aligned_words/`

**Key features:**
- Text normalization for common backchannel variations
- Lookahead logic to skip short matches in favor of longer sequences
- Speaker flip detection based on alignment quality (match rate < 50% threshold)
- Processes all 1,155 conversations with both SWDA and transcript data

**Output:** `aligned_words/aligned_conv_{conv_num}.csv` containing:
- `speaker`: A or B
- `alignment_type`: match, mismatch, insertion, or deletion
- `swda_word`: word from SWDA transcript
- `trans_word`: word from MS-State transcript
- `utterance_index`, `subutterance_index`: SWDA utterance IDs
- `act_tag`: Dialog act tag
- `conv_no`: Conversation number
- `turn_id`: MS-State turn ID
- `start`, `end`: Word timing in seconds

### 2. Turn-Level Aggregation (`process_aligned.py`)

Post-processes word-level alignments to create turn-level utterances:
- Drops deletion alignments (words only in SWDA, not in transcript)
- Sorts by timestamp
- Fills missing act tags using forward/backward/nearest neighbor passes
- Groups words by utterance to reconstruct full turns
- Outputs turn-level data to `aligned_turns/`

**Output:** `aligned_turns/aligned_turns_{conv_num}.csv` containing:
- `speaker`: A or B
- `utterance_index`, `subutterance_index`: SWDA utterance IDs
- `start`, `end`: Utterance timing in seconds
- `turn_id`: MS-State turn ID
- `transcript_swda`: Full utterance text from SWDA
- `transcript_ms`: Full utterance text from MS-State
- `act_tag`: Dialog act tag (simplified DAMSL tags)

## Usage

### Setup

```bash
# Clone and setup SWDA data
bash download.sh

# Install dependencies
pip install -r requirements.txt
```

### Run Alignment

```bash
# Step 1: Word-level alignment
python align_swda_mstrans.py

# Step 2: Turn-level aggregation
python process_aligned.py
```

### Processing Time

- Word-level alignment: ~5 minutes on Intel Core Ultra 7 155U
- Turn-level aggregation: ~3 minutes on Intel Core Ultra 7 155U

## Results

- **1,155 conversations** successfully aligned
- **84 conversations** had speaker channels flipped (automatically corrected)
- **~99.7% word coverage** on average (missing act_tags tracked in logs)
- Output files sorted by conversation number

### Alignment Quality

Most conversations have >99% of words successfully aligned with act tags. Conversations with most unfilled act_tags:
- 2768: 46/2121 words (2.2%)
- 2884: 45/2377 words (1.9%)
- 2386: 42/2215 words (1.9%)

See `logs.txt` for complete processing details.

## Files

- `align_swda_mstrans.py`: Main word-level alignment script
- `process_aligned.py`: Turn-level aggregation script
- `compare_alignment.py`: Comparison of alignment algorithms (difflib's Ratcliff-Obershelp algorithm vs. Needleman–Wunsch algorithm)
- `download.sh`: Setup script for SWDA data
- `flipped_channels.txt`: List of conversations with flipped speaker channels
- `logs.txt`: Processing logs and statistics
- `aligned_words/`: Word-level alignment outputs (1,155 files)
- `aligned_turns/`: Turn-level aggregated outputs (1,155 files)

## License

See `LICENSE` file for details.

## References

- [Switchboard Dialog Act Corpus](https://github.com/cgpotts/swda)
- [DAMSL Annotation Manual](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html)
- [Mississippi State Switchboard resegmentation project](https://isip.piconepress.com/projects/switchboard/)
