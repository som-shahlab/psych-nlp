import copy
import pytest
import re
import sys

sys.path.append("../psynlp")

import features
from features import parse
from features.utils import extract_metadata_from_line
from features.utils import extract_metadata_from_path
from features.utils import preprocess_text
from features.utils import generate_line_id
from features.utils import min_sec_fmt
from features.featurizers import Line
from features.featurizers import Transcript


line0 = Line(speaker="P", text="line zero", start_time=0.0)
line1 = Line(speaker="T", text="line one", start_time=1.0)
line2 = Line(speaker="T", text="line two", start_time=2.0)
line3 = Line(speaker="P", text="line three", start_time=3.0)
line4 = Line(speaker="P", text="", start_time=4.0)
line5 = Line(speaker="T", text="line five", start_time=5.0)

transcript = Transcript(
    session_id="012345", lines=[line0, line1, line2, line3, line4, line5]
)


def test_drop_blank_lines():
    tmp_transcript = copy.deepcopy(transcript)
    tmp_transcript.drop_blank_lines()
    assert tmp_transcript.lines[3].text == "line three"
    assert tmp_transcript.lines[4].text == "line five"
    assert len(transcript.lines) == 6


def test_merge_repeat_speaker_lines():
    tmp_transcript = copy.deepcopy(transcript)
    tmp_transcript.merge_repeat_speaker_lines()
    assert tmp_transcript.lines[1].text == "line one line two"
    assert tmp_transcript.lines[2].text == "line three"


def test_impute_end_times():
    tmp_transcript = copy.deepcopy(transcript)
    tmp_transcript.impute_end_times()
    assert tmp_transcript.lines[0].end_time == 1.0
    assert tmp_transcript.lines[-1].end_time is None


def test_assign_line_ids():
    tmp_transcript = copy.deepcopy(transcript)
    tmp_transcript.assign_line_ids()
    assert tmp_transcript.lines[0].line_id == tmp_transcript.session_id + "_" + "000000"
    assert tmp_transcript.lines[5].line_id == tmp_transcript.session_id + "_" + "000005"


path_to_LIWC = "../psynlp/features/lexicons/LIWC2007_English100131.dic"


def test_past_featurizer():
    line_past = Line(text="became didn't spoken she'll")
    featurizer = features.featurizers.LIWCFeaturizer(
        "past_oriented", "past", path_to_lexicon=path_to_LIWC
    )
    line_past.calculate_features([featurizer])
    assert line_past.features["past_oriented"] == 3


def test_present_featurizer():
    line_present = Line(text="admit ain't you're admit")
    featurizer = features.featurizers.LIWCFeaturizer(
        "present_oriented", "present", path_to_lexicon=path_to_LIWC
    )
    line_present.calculate_features([featurizer])
    assert line_present.features["present_oriented"] == 4


def test_future_featurizer():
    line_future = Line(text="we'll will may")
    featurizer = features.featurizers.LIWCFeaturizer(
        "future_oriented", "future", path_to_lexicon=path_to_LIWC
    )
    line_future.calculate_features([featurizer])
    assert line_future.features["future_oriented"] == 3


def test_extract_metadata_from_line_expected_fmt():
    # Line is formatted as expected
    res = extract_metadata_from_line("John [TIME: 20:15]: I'm sad")
    person, time_in_mins, line = res
    assert person == "John"
    assert time_in_mins == pytest.approx(20.25)
    assert line == " I'm sad"  # Note: content includes leading space


def test_extract_metadata_from_line_missing_person_id():
    # Line is missing a person identifier
    res = extract_metadata_from_line("[TIME: 20:13]: I'm sad")
    assert res is None


def test_extract_metadata_from_line_missing_timestamp():
    # Line is missing time stamp
    res = extract_metadata_from_line("John: I'm sad")
    assert res is None


def test_extract_metadata_from_line_wrong_fmt():
    # A sensible but unexpected format
    res = extract_metadata_from_line("John (20:13) I'm sad")
    assert res is None


def test_extract_metadata_from_path_expected_fmt():
    res = extract_metadata_from_path("S7_060504_P1_03.02.01_A.TXT")
    session_id, session_num = res
    assert session_id == "060504"
    assert session_num == 7


def test_extract_metadata_from_path_modified_valid_fmt():
    # We may have cases where the leading "0"s are added/dropped
    res = extract_metadata_from_path("S07_60504_P1_03.02.01_A.TXT")
    session_id, session_num = res
    assert session_id == "060504"
    assert session_num == 7


def test_preprocess_text_keep_single_quote():
    assert preprocess_text("Don't") == "don't"


def test_preprocess_text_remove_punctuations():
    assert preprocess_text("Funny! [LAUGHING]. Wow. [") == "funny wow"


def test_remove_multiple_bracketed_expressions():
    assert (
        preprocess_text("[first] this [second] is [third] complete [fourth].")
        == "this is complete"
    )


def test_generate_line_id():
    assert (
        generate_line_id(sess_id="080102", line_num=7, line_id_len=6) == "080102_000007"
    )


def test_min_sec_fmt_general():
    assert min_sec_fmt(10.75) == "10:45"
    assert min_sec_fmt(10.9) == "10:54"
    assert min_sec_fmt(10.99) == "10:59"


def test_min_sec_fmt_more_than_60_minutes():
    assert min_sec_fmt(60.25) == "60:15"


def test_min_sec_fmt_zero():
    assert min_sec_fmt(0.00) == "00:00"
