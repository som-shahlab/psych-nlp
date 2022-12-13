import math
import re

from typing import Optional
from typing import Tuple
from typing import Iterable

def min_sec_fmt(minutes: float) -> str:
    minutes_rounded = math.floor(minutes)
    seconds = (minutes - minutes_rounded) * 60
    seconds = round(seconds)
    # import pdb
    # pdb.set_trace()
    return f"{minutes_rounded:02d}:{seconds:02d}"
    
    
def preprocess_text(text: str) -> str:
    # Make everything lower case
    cleaned = text.lower()

    # Remove any meta-comments added by annotators
    # e.g., "Haha [LAUGHTER] funny!" -> "Haha funny!"
    cleaned = re.sub(r"\[.*?\]", "", cleaned)

    # Remove any punctuations (except single apostrophes, as in can't or don't)
    punc_table = str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[]^_`{|}~')
    cleaned = cleaned.translate(punc_table)

    # Get rid of any extraneous space in between words
    cleaned = " ".join([w for w in cleaned.split()])
    return cleaned


def generate_line_id(sess_id: str, line_num: int, line_id_len: int) -> str:
    return sess_id + "_" + str(line_num).zfill(line_id_len)


def extract_metadata_from_path(
    path_to_transcript: str, num_chars_in_id: int = 6, verbose: int = 0
) -> Optional[Tuple[str, int]]:
    """Extract the Session ID from the transcript path"""
    pattern = r"S([0-9]+)_([0-9]+)_.*"  # Edit this line for your dataset
    pattern_match = re.search(pattern, path_to_transcript)
    if pattern_match is None:
        if verbose:
            print(f"Couldn't extract Session ID/num from {path_to_transcript}")
        return None
    session_num = int(pattern_match.group(1))  # the numbers following 'S'
    session_id: str = pattern_match.group(2)  # The first numbers after '_'
    session_id = session_id.zfill(num_chars_in_id)
    return session_id, session_num


def extract_metadata_from_line(
        line: str, verbose: int = 0
    ) -> Optional[Tuple[str, float, str]]:
        """Extract the timestamp of a given talk turn
        
        Note that this is dataset specific. The user will need to specify this
        in order for the overall parsing to work on their own dataset.
        While the timestamp is typically not discernible from the audio
        itself, it's possible to use machine learning to identify the speaker
        in cases where source may be ambiguous. Because we're using
        human-generated annotations, we just use the speaker identity given.
        
        Args:
            line: A text string containing the line from which the metadata
                should be extracted

        Returns:
            person: String of the person's name that appears first in the line
            time_in_mins: Float indicating the total time (converted to minutes)
                indicated by the time stamp.
            line_content: String of text with person's name and
                timestamp extracted/removed
        """
        # Our pattern is e.g., "P [TIME: 20:15]: I'm sad"
        # which means the patient said "I'm sad" 20'15" into the conversation.
        pattern = r"([A-Za-z]+)\s\[TIME: ([0-9]+):([0-9]+)\]:(.*)"  # Edit as needed
        pattern_match = re.search(pattern, line)
        if pattern_match is None:
            if verbose:
                print(f"Could not parse this line:\n\t'{line}'\n")
            return None
        person: str = pattern_match.group(1)
        time_in_secs: float = int(pattern_match.group(2)) * 60.0 + int(
            pattern_match.group(3)
        )
        time_in_mins: float = time_in_secs / 60.0
        line_content: str = pattern_match.group(4)
        return person, time_in_mins, line_content
    
    
# Currently O(T * N) where T is the number in the set, N is length of line
def count_terms_in_line(text: str, terms: Iterable[str]):
    num_occur_terms_in_line = 0
    for term in terms:
        # TODO: Account for the fact that some terms represent two words rather
        # than just one word, so if you're normalizing by word count you should
        # probably count not just `1` here but len(term.split())
        num_occur_terms_in_line += sum(
            1 for _ in re.finditer(r"\b%s\b" % re.escape(term), text, re.IGNORECASE)
        )
    return num_occur_terms_in_line

