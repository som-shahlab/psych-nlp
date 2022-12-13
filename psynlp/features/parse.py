import argparse
import os
import sys

sys.path.append("../../psynlp")

import pandas as pd
import pickle
from tqdm import tqdm

from features import config
from features import featurizers
from features import utils


def parse_transcript(path_to_transcript, featurizer_objs):
    # Make sure we can extract the necessary metadata from the
    # transcript path before parsing its contents
    path_metadata = utils.extract_metadata_from_path(path_to_transcript)
    if path_metadata is None:
        return None
    session_id, session_num = path_metadata
    transcript_obj = featurizers.Transcript(
        session_id=session_id, session_num=session_num, fpath=path_to_transcript
    )

    with open(path_to_transcript, mode="r", encoding="utf-8") as f:
        for line in f:
            line_metadata = utils.extract_metadata_from_line(line)
            if line_metadata is None:
                continue
            person, time_in_mins, spoken_line = line_metadata
            cleaned_line = utils.preprocess_text(spoken_line)

            line_obj = featurizers.Line()
            line_obj.speaker = person
            line_obj.start_time = time_in_mins
            line_obj.text = cleaned_line

            transcript_obj.lines.append(line_obj)

    transcript_obj.drop_blank_lines()
    transcript_obj.merge_repeat_speaker_lines()
    transcript_obj.impute_end_times()
    transcript_obj.assign_line_ids()
    transcript_obj.calculate_features(featurizer_objs)

    return transcript_obj


if __name__ == "__main__":
    # For each transcript in the directory
    #   1. Parse the transcript to create a transcript object
    #   2. Serialize the transcript object
    # Then write the serialized transcript objects to disk

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="transcripts.tsv",
        help="Location where the .tsv containing the summary"
        " of transcript preprocessing & featurization should be saved",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="If True, cache transcript objects as pickle "
        "prior to serialization serialization",
    )
    parser.add_argument(
        "--cache_filepath",
        type=str,
        default="transcripts.pkl",
        help="Location where to cache a pickle file"
        "representing transcripts before serialization",
    )
    args = parser.parse_args()

    featurizer_objs = [
        featurizers.LIWCFeaturizer("you_pronouns", "you"),
        featurizers.LIWCFeaturizer("they_pronouns", "they"),
        featurizers.LIWCFeaturizer("personal_pronouns", "ppron"),
        featurizers.LIWCFeaturizer("i_pronouns", "i"),
        featurizers.LIWCFeaturizer("we_pronouns", "we"),
        featurizers.LIWCFeaturizer("past_oriented", "past"),
        featurizers.LIWCFeaturizer("present_oriented", "present"),
        featurizers.LIWCFeaturizer("future_oriented", "future"),
        featurizers.EmoLexFeaturizer("negative", "negative"),
        featurizers.EmoLexFeaturizer("positive", "positive"),
        featurizers.CheckingForUnderstandingFeaturizer(),
        featurizers.DemonstratingUnderstandingFeaturizer(),
        featurizers.HedgingFeaturizer(),
        featurizers.AbsolutistFeaturizer(),
        featurizers.SecondsPerTalkTurnFeaturizer(),
        featurizers.WordsPerSecondFeaturizer(),
    ]

    meta_df = pd.read_csv(
        config.METADATA_PATH,
        sep="\t",
        converters={
            "ID_number": str,
            "Site_ID_number": str,
            "Therapist_ID_number": str,
            "Patient_ID_number": str,
        },
    )

    print(f"Processing {len(meta_df)} transcripts...")

    # If the transcripts have already been cached, load from disk
    # Otherwise, preprocess + featurize each transcript individually
    if os.path.exists(args.cache_filepath) and args.use_cache:
        with open(args.cache_filepath, "rb") as f:
            transcripts = pickle.load(f)
    else:
        transcripts = []
        for i, row in tqdm(meta_df.iterrows(), total=meta_df.shape[0]):
            path_to_transcript = row["gold_path"]
            transcripts.append(parse_transcript(path_to_transcript, featurizer_objs))
        if args.use_cache:
            with open(args.cache_filepath, "wb") as f:
                pickle.dump(transcripts, f)

    # Serialize all the transcripts and store them as a .tsv
    tsv_str = ""
    for i, transcript in enumerate(transcripts):
        if i == 0:
            tsv_str += transcript.to_tsv(use_header=True)
        else:
            tsv_str += transcript.to_tsv(use_header=False)
    with open("transcripts.tsv", "w") as f:
        f.write(tsv_str)
