import re
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
import time

from tqdm import tqdm
import numpy as np

import features
from features import config
from features import liwc
from features import utils


class Featurizer(object):
    def __init__(self, feature_descr: Optional[str] = None):
        self.feature_descr = feature_descr

    def featurize(self) -> Tuple[Union[float, int], str]:
        return 0, ""


class Line(object):
    def __init__(
        self,
        line_id: Optional[str] = None,
        speaker: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        text: Optional[str] = None,
        features: Optional[Mapping[str, float]] = None,
    ):
        self.line_id = line_id
        self.speaker = speaker
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.features = features if features is not None else dict()

    def calculate_features(self, featurizer_objs: Iterable[Featurizer]):
        for featurizer_obj in featurizer_objs:
            feat_value, feat_descr = featurizer_obj.featurize(self)
            self.features[feat_descr] = feat_value

    def __str__(self):
        p_str = f"{self.line_id} " if self.line_id is not None else ""
        p_str += f"{self.speaker} " if self.speaker is not None else ""
        p_str += (
            f"[TIME: {features.utils.min_sec_fmt(self.start_time)}] "
            if self.start_time is not None
            else ""
        )
        p_str += f"{self.text}"
        p_str += (
            f" [TIME: {features.utils.min_sec_fmt(self.end_time)}]"
            if self.end_time is not None
            else ""
        )
        return p_str


class Transcript(object):
    def __init__(
        self,
        lines: Optional[List[Line]] = None,
        session_id: Optional[str] = None,
        session_num: Optional[int] = None,
        fpath: Optional[str] = None,
    ):
        self.lines = lines if lines is not None else []
        self.session_id = session_id
        self.session_num = session_num
        self.fpath = fpath

    def postprocess(self):
        self.drop_blank_lines()
        self.merge_repeat_speaker_lines()
        self.impute_end_times()
        self.assign_line_ids()

    def drop_blank_lines(self):
        new_list = []
        for line in self.lines:
            if len(line.text) > 0:
                new_list.append(line)
        self.lines = new_list

    def merge_repeat_speaker_lines(self):
        """Merge two lines if they are adjacent and from the same speaker
        NOTE: Does not handle line_id or features
        Assumes text and num_words has been initialized
        """
        prev_line = self.lines[0]
        new_list = [prev_line]
        for curr_line in self.lines[1:]:
            if curr_line.speaker == prev_line.speaker:
                new_text = prev_line.text + " " + curr_line.text
                new_text = new_text.strip()
                new_line = Line(
                    line_id=None,
                    speaker=prev_line.speaker,
                    start_time=prev_line.start_time,
                    end_time=curr_line.end_time,
                    text=new_text,
                )
                new_list.pop()
                new_list.append(new_line)
            else:
                new_list.append(curr_line)
            prev_line = curr_line
        self.lines = new_list

    def impute_end_times(self):
        """
        Note: The last sentence will not have an imputed end time. Just have to drop it...
        """
        prev_line = self.lines[0]
        for curr_line in self.lines[1:]:
            prev_line.end_time = curr_line.start_time
            prev_line = curr_line

    def assign_line_ids(self):
        for i, line in enumerate(self.lines):
            line.line_id = features.utils.generate_line_id(
                self.session_id, i, line_id_len=6
            )

    def calculate_features(self, featurizer_objs: Iterable[Featurizer]):
        for line in tqdm(self.lines, total=len(self.lines)):
            line.calculate_features(featurizer_objs)

    def to_tsv(self, fpath: str = "transcripts.csv", use_header: bool = False):
        serialized = ""
        if use_header:
            header_elements = [
                "session_id",
                "line_id",
                "speaker",
                "start_time",
                "end_time",
                "feature_descr",
                "feature_value",
                "text",
            ]
            header = "\t".join(header_elements)
            serialized += header + "\n"

        for transcript_line in self.lines:
            for feat_descr, feat_value in transcript_line.features.items():
                tsv_line = []
                tsv_line.append(str(self.session_id))
                tsv_line.append(transcript_line.line_id)
                tsv_line.append(transcript_line.speaker)
                tsv_line.append(str(transcript_line.start_time))
                if transcript_line.end_time is not None:
                    tsv_line.append(str(transcript_line.end_time))
                tsv_line.append(feat_descr)
                tsv_line.append(str(feat_value))
                tsv_line.append(transcript_line.text)
                serialized += "\t".join(tsv_line) + "\n"

        return serialized

    def __str__(self):
        return "\n".join([str(l) for l in self.lines])


#################################################
# LIWC FEATURIZERS (PRONOUNS, TIME ORIENTATION) #
#################################################


class LIWCFeaturizer(Featurizer):
    def __init__(
        self,
        feature_descr: str,
        target_category: str,
        path_to_lexicon: str = config.LIWC_PATH,
    ):
        self.feature_descr = feature_descr
        self.target_category = target_category
        self.liwc_obj = liwc.LIWC(path_to_lexicon)

    def featurize(self, line: Line):
        line_ctr = self.liwc_obj.parse(line.text.lower().split())
        return line_ctr[self.target_category], self.feature_descr


###########################################
# EMOLEX FEATURIZERS (EMOTIONAL POLARITY) #
###########################################


class EmoLexFeaturizer(Featurizer):
    def __init__(
        self,
        feature_descr: str,
        target_emo: str,
        path_to_lexicon: str = config.EMOLEX_PATH,
    ):
        self.feature_descr = feature_descr
        self.target_set = set()
        with open(path_to_lexicon, mode="r", encoding="utf-8") as f:
            for line in f:
                word, emo, word_conveys_emo = line.split()
                if emo == target_emo and word_conveys_emo == "1":
                    self.target_set.add(word)

    def featurize(self, line: Line):
        n_terms_in_line = features.utils.count_terms_in_line(line.text, self.target_set)
        return n_terms_in_line, self.feature_descr


#####################
# THERAPIST TACTICS #
#####################


class CheckingForUnderstandingFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "checking_for_understanding"):
        self.feature_descr = feature_descr
        self.target_set = [
            "it sounds like",
            "you seem to be saying",
            "let me make sure",
            "heard you correctly",
            "let me see",
            "sounds like",
            "seems like",
            "it seems",
            "it sounds",
            "that seems",
            "that sounds",
            "this seems",
            "this sounds",
            "you seem",
            "you sound",
        ]

    def featurize(self, line: Line):
        n_terms_in_line = features.utils.count_terms_in_line(line.text, self.target_set)
        return n_terms_in_line, self.feature_descr


class DemonstratingUnderstandingFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "demonstrating_understanding"):
        self.feature_descr = feature_descr
        self.target_set = [
            "i hear you",
            "i see",
            "i understand",
            "i can see",
            "i get that",
            "gotcha",
        ]

    def featurize(self, line: Line):
        n_terms_in_line = features.utils.count_terms_in_line(line.text, self.target_set)
        return n_terms_in_line, self.feature_descr


class HedgingFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "hedging"):
        self.feature_descr = feature_descr
        self.target_set = [
            "think",
            "thought",
            "thinking",
            "almost",
            "apparent",
            "apparently",
            "appear",
            "appeared",
            "appears",
            "approximately",
            "around",
            "assume",
            "assumed",
            "certain amount",
            "certain extent",
            "certain level",
            "claim",
            "claimed",
            "doubt",
            "doubtful",
            "essentially",
            "estimate",
            "estimated",
            "feel",
            "felt",
            "frequently",
            "from our perspective",
            "generally",
            "guess",
            "in general",
            "in most cases",
            "in most instances",
            "in our view",
            "indicate",
            "indicated",
            "largely",
            "likely",
            "mainly",
            "may",
            "maybe",
            "might",
            "mostly",
            "often",
            "on the whole",
            "ought",
            "perhaps",
            "plausible",
            "plausibly",
            "possible",
            "possibly",
            "postulate",
            "postulated",
            "presumable",
            "probable",
            "probably",
            "relatively",
            "roughly",
            "seems",
            "should",
            "sometimes",
            "somewhat",
            "suggest",
            "suggested",
            "suppose",
            "suspect",
            "tend to",
            "tends to",
            "typical",
            "typically",
            "uncertain",
            "uncertainly",
            "unclear",
            "unclearly",
            "unlikely",
            "usually",
            "broadly",
            "tended to",
            "presumably",
            "suggests",
            "from this perspective",
            "from my perspective",
            "in my view",
            "in this view",
            "in our opinion",
            "in my opinion",
            "to my knowledge",
            "fairly",
            "quite",
            "rather",
            "argue",
            "argues",
            "argued",
            "claims",
            "feels",
            "indicates",
            "supposed",
            "supposes",
            "suspects",
            "postulates",
        ]

    def featurize(self, line: Line):
        n_terms_in_line = features.utils.count_terms_in_line(line.text, self.target_set)
        return n_terms_in_line, self.feature_descr


class AbsolutistFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "absolutist"):
        self.feature_descr = feature_descr
        self.target_set = [
            "absolutely",
            "all",
            "always",
            "complete",
            "completely",
            "constant",
            "constantly",
            "definitely",
            "entire",
            "ever",
            "every",
            "everyone",
            "everything",
            "full",
            "must",
            "never",
            "nothing",
            "totally",
            "whole",
        ]

    def featurize(self, line: Line):
        n_terms_in_line = features.utils.count_terms_in_line(line.text, self.target_set)
        return n_terms_in_line, self.feature_descr


class SecondsPerTalkTurnFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "seconds_per_talk_turn"):
        self.feature_descr = feature_descr

    def featurize(self, line: Line):
        if line.end_time is None or line.start_time is None:
            return None, self.feature_descr
        return (line.end_time - line.start_time) * 60.0, self.feature_descr


class WordsPerSecondFeaturizer(Featurizer):
    def __init__(self, feature_descr: str = "words_per_second"):
        self.feature_descr = feature_descr

    def featurize(self, line: Line):  # TODO: Handle case where line has duration 0
        wps = None
        if line.end_time is None or line.start_time is None:
            return wps, self.feature_descr
        line_dur_in_seconds = (line.end_time - line.start_time) * 60.0
        num_words_in_line = len(line.text.split())
        if line_dur_in_seconds > 0 and num_words_in_line > 0:
            wps = num_words_in_line / line_dur_in_seconds
        return wps, self.feature_descr
