"""Golden questions for the CD/DVD player manual (resources/user-manual-cd-player.pdf).

Questions are deliberately paraphrased (e.g. "freeze the picture" for pause) so
they test semantic retrieval, not keyword matching.

- `answer_phrase`: exact text from the manual that answers the question; the
  chunk containing it should be retrieved (free check).
- `must_match`: regexes (case-insensitive) the answer must contain; cheap
  stand-in for an LLM judge on the key facts (free check).
- `expected_output`: reference answer for the LLM judge.
- `judged`: included in the paid LLM-judge tier (keep this subset small).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_output: str
    answer_phrase: str = ""            # empty for out-of-scope questions
    must_match: tuple[str, ...] = ()
    judged: bool = False


IN_SCOPE = [
    GoldenCase(
        "region", "Which region is this DVD player for?",
        "The player is region 2; it only plays DVD video discs marked 2 or ALL.",
        "region number of this player is 2",
        must_match=(r"\b2\b",)),
    GoldenCase(
        "parental-password", "How many digits is the parental lock password?",
        "It is a 5-digit security code; the default is 99999 when using the player for the first time.",
        "personal 5-digit",
        must_match=(r"\b(5|five)[- ]digit",),
        judged=True),
    GoldenCase(
        "screen-saver", "When does the screen saver come on?",
        "After 20 minutes with the power on and no disc in the player, or with the disc stopped, "
        "if the Screen saver setting is On. Pressing any button turns it off.",
        "If 20 minutes elapse",
        must_match=(r"\b(20|twenty) minutes",)),
    GoldenCase(
        "pause", "How do I freeze the picture?",
        "Press PAUSE/STEP during playback; the picture becomes still and the sound is muted. "
        "Press PLAY to resume normal playback.",
        "Press PAUSE/STEP during playback",
        must_match=(r"PAUSE/STEP",),
        judged=True),
    GoldenCase(
        "resume", "Will it continue from where I stopped the disc last time?",
        "The resume playback function memorizes where playback stopped, and pressing PLAY "
        "starts playback from that location.",
        "memorizes the location where playback is stopped",
        must_match=(r"\bPLAY\b",)),
    GoldenCase(
        "cleaning", "How should I clean the device?",
        "Clean it only with a dry cloth.",
        "Clean only with dry cloth",
        must_match=(r"dry cloth",),
        judged=True),
    GoldenCase(
        "headphone-volume", "Is loud headphone listening safe?",
        "Keep headphone volume at a moderate level; continuous use at high volume may damage hearing.",
        "keep the volume at a moderate level",
        must_match=(r"moderate",)),
    GoldenCase(
        "zoom", "How can I magnify part of the image?",
        "Press ZOOM during playback, then select the zoom point; press ZOOM repeatedly to change "
        "the magnification level.",
        "Press ZOOM during playback",
        must_match=(r"\bZOOM\b",)),
    GoldenCase(
        "shuffle", "How do I shuffle tracks?",
        "During playback, press RANDOM while pressing SHIFT to show 'Shuffle On'; pressing it again "
        "switches between Shuffle On and Shuffle off.",
        "press RANDOM while pressing SHIFT",
        must_match=(r"\bRANDOM\b", r"\bSHIFT\b")),
    GoldenCase(
        "subtitles", "How do I change the subtitle language?",
        "Press SUBTITLE during playback to show the current setting, then press SUBTITLE again "
        "repeatedly to select from the subtitle languages on the DVD.",
        "Press SUBTITLE during playback",
        must_match=(r"\bSUBTITLE\b",),
        judged=True),
    GoldenCase(
        "bookmark", "How do I jump to a saved bookmark?",
        "Register a bookmark first. Then during playback press the display button repeatedly to show "
        "'Bookmark', select the bookmark and press ENTER; playback starts from it.",
        "First register a bookmark",
        must_match=(r"\bENTER\b",)),
    GoldenCase(
        "repeat", "How do I loop the current chapter?",
        "During playback, press REPEAT repeatedly while pressing SHIFT to select the repeat mode "
        "(title, chapter or track).",
        "press REPEAT repeatedly while pressing SHIFT",
        must_match=(r"\bREPEAT\b", r"\bSHIFT\b")),
]

# Questions the manual cannot answer: the app should say it does not know.
OUT_OF_SCOPE = [
    GoldenCase("world-cup", "Who won the 1998 FIFA World Cup?",
               "I do not know; the documents don't cover this."),
    GoldenCase("price", "How much does this player cost in US dollars?",
               "I do not know; the documents don't cover this."),
]

# Retrieval misses we know about; xfail so the suite stays useful, and an
# XPASS shows up when a change fixes them.
KNOWN_RETRIEVAL_MISSES = {
    "pause": "paraphrase 'freeze the picture' doesn't retrieve the PAUSE/STEP chunk in the "
             "top 5 (rank ~8); expected to improve with page-based chunking",
}
