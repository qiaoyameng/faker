import unicodedata
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple, cast

from ...exceptions import BaseFakerException
from .. import BaseProvider

localized = True

# 'Latin' is the default locale
default_locale = "la"


class ExactByteLengthException(BaseFakerException):
    """Base class for errors raised by |text_with_exact_byte_length|."""


class UnsatisfiableByteLength(ExactByteLengthException):
    """No sequence of the supplied fragments reaches the requested byte size.

    This is a mathematical impossibility for the given fragment/separator
    combination, not a resource problem: the search enumerated every byte
    size that can be produced and none of them equals the target.
    """


class ByteLengthBudgetExhausted(ExactByteLengthException):
    """The operation budget ran out before the search could finish.

    Distinct from |UnsatisfiableByteLength|: the answer is unknown here,
    because the search space was not fully explored.
    """


_NORMALIZATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def _interactive_tail(normalized_text: str) -> str:
    """Return the suffix of an already-normalized string that can interact
    with text appended to it during a later normalization.

    Canonical reordering and composition can never cross a character with
    canonical combining class 0 (a "starter"), so everything from the last
    starter on is all the context an appended string may need. A string
    without starters can interact as a whole.
    """
    for index in range(len(normalized_text) - 1, -1, -1):
        if unicodedata.combining(normalized_text[index]) == 0:
            return normalized_text[index:]
    return normalized_text


class Provider(BaseProvider):
    """Implement default lorem provider for Faker.

    .. important::
       The default locale of the lorem provider is ``la``. When using a locale
       without a localized lorem provider, the ``la`` lorem provider will be
       used, so generated words will be in pseudo-Latin. The locale used for
       the standard provider docs was ``en_US``, and ``en_US`` has a localized
       lorem provider which is why the samples here show words in American
       English.
    """

    word_connector = " "
    sentence_punctuation = "."

    def get_words_list(
        self,
        part_of_speech: Optional[str] = None,
        ext_word_list: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Get list of words.

        ``ext_word_list`` is a parameter that allows the user to provide a list
        of words to be used instead of the built-in word list. If ``ext_word_list``
        is provided, then the value of ``part_of_speech`` is ignored.

        ``part_of_speech`` is a parameter that defines to what part of speech
        the returned word belongs. If ``ext_word_list`` is not ``None``, then
        ``part_of_speech`` is ignored. If the value of ``part_of_speech`` does
        not correspond to an existent part of speech according to the set locale,
        then an exception is raised.

        :sample: part_of_speech="abc", ext_word_list=['abc', 'def', 'ghi', 'jkl']
        :sample: part_of_speech="abc"
        :sample: ext_word_list=['abc', 'def', 'ghi', 'jkl']

        .. warning::
            Depending on the length of a locale provider's built-in word list or
            on the length of ``ext_word_list`` if provided, a large ``nb`` can
            exhaust said lists if ``unique`` is ``True``, raising an exception.
        """

        if ext_word_list is not None:
            word_list = ext_word_list
        elif part_of_speech:
            if part_of_speech not in self.parts_of_speech:  # type: ignore[attr-defined]
                raise ValueError(f"{part_of_speech} is not recognized as a part of speech.")
            else:
                word_list = self.parts_of_speech[part_of_speech]  # type: ignore[attr-defined]
        else:
            word_list = self.word_list  # type: ignore[attr-defined]

        return list(word_list)

    def words(
        self,
        nb: int = 3,
        ext_word_list: Optional[List[str]] = None,
        part_of_speech: Optional[str] = None,
        unique: bool = False,
    ) -> List[str]:
        """Generate a tuple of words.

        The ``nb`` argument controls the number of words in the resulting list,
        and if ``ext_word_list`` is provided, words from that list will be used
        instead of those from the locale provider's built-in word list.

        if ``word_list`` is not provided, the method will use a default value of None,
        which will result in the method calling the ``get_words_list`` method to get the
        word list. If ``word_list`` is provided, the method will use the provided list.

        If ``unique`` is ``True``, this method will return a list containing
        unique words. Under the hood, |random_sample| will be used for sampling
        without replacement. If ``unique`` is ``False``, |random_choices| is
        used instead, and the list returned may contain duplicates.

        :sample:
        :sample: nb=5
        :sample: nb=5, ext_word_list=['abc', 'def', 'ghi', 'jkl']
        :sample: nb=4, ext_word_list=['abc', 'def', 'ghi', 'jkl'], unique=True
        """

        word_list = self.get_words_list(part_of_speech=part_of_speech, ext_word_list=ext_word_list)

        if unique:
            unique_samples = cast(List[str], self.random_sample(word_list, length=nb))
            return unique_samples
        samples = cast(List[str], self.random_choices(word_list, length=nb))
        return samples

    def word(self, part_of_speech: Optional[str] = None, ext_word_list: Optional[Sequence[str]] = None) -> str:
        """Generate a word.

        This method uses |words| under the hood with the ``nb`` argument set to
        ``1`` to generate the result.

        :sample:
        :sample: ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        word_list = self.get_words_list(part_of_speech, ext_word_list)

        return self.words(1, word_list)[0]

    def sentence(
        self, nb_words: int = 6, variable_nb_words: bool = True, ext_word_list: Optional[Sequence[str]] = None
    ) -> str:
        """Generate a sentence.

        The ``nb_words`` argument controls how many words the sentence will
        contain, and setting ``variable_nb_words`` to ``False`` will generate
        the exact amount, while setting it to ``True`` (default) will generate
        a random amount (+/-40%, minimum of 1) using |randomize_nb_elements|.

        Under the hood, |words| is used to generate the words, so the argument
        ``ext_word_list`` works in the same way here as it would in that method.

        :sample: nb_words=10
        :sample: nb_words=10, variable_nb_words=False
        :sample: nb_words=10, ext_word_list=['abc', 'def', 'ghi', 'jkl']
        :sample: nb_words=10, variable_nb_words=True,
                 ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        if nb_words <= 0:
            return ""

        if variable_nb_words:
            nb_words = self.randomize_nb_elements(nb_words, min=1)

        word_list = self.get_words_list(ext_word_list=ext_word_list)
        words = list(self.words(nb=nb_words, ext_word_list=word_list))
        words[0] = words[0].title()

        return self.word_connector.join(words) + self.sentence_punctuation

    def sentences(self, nb: int = 3, ext_word_list: Optional[Sequence[str]] = None) -> List[str]:
        """Generate a list of sentences.

        This method uses |sentence| under the hood to generate sentences, and
        the ``nb`` argument controls exactly how many sentences the list will
        contain. The ``ext_word_list`` argument works in exactly the same way
        as well.

        :sample:
        :sample: nb=5
        :sample: nb=5, ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        return [self.sentence(ext_word_list=ext_word_list) for _ in range(0, nb)]

    def paragraph(
        self, nb_sentences: int = 3, variable_nb_sentences: bool = True, ext_word_list: Optional[Sequence[str]] = None
    ) -> str:
        """Generate a paragraph.

        The ``nb_sentences`` argument controls how many sentences the paragraph
        will contain, and setting ``variable_nb_sentences`` to ``False`` will
        generate the exact amount, while setting it to ``True`` (default) will
        generate a random amount (+/-40%, minimum of 1) using
        |randomize_nb_elements|.

        Under the hood, |sentences| is used to generate the sentences, so the
        argument ``ext_word_list`` works in the same way here as it would in
        that method.

        :sample: nb_sentences=5
        :sample: nb_sentences=5, variable_nb_sentences=False
        :sample: nb_sentences=5, ext_word_list=['abc', 'def', 'ghi', 'jkl']
        :sample: nb_sentences=5, variable_nb_sentences=False,
                 ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        if nb_sentences <= 0:
            return ""

        if variable_nb_sentences:
            nb_sentences = self.randomize_nb_elements(nb_sentences, min=1)

        para = self.word_connector.join(self.sentences(nb_sentences, ext_word_list=ext_word_list))

        return para

    def paragraphs(self, nb: int = 3, ext_word_list: Optional[Sequence[str]] = None) -> List[str]:
        """Generate a list of paragraphs.

        This method uses |paragraph| under the hood to generate paragraphs, and
        the ``nb`` argument controls exactly how many sentences the list will
        contain. The ``ext_word_list`` argument works in exactly the same way
        as well.

        :sample: nb=5
        :sample: nb=5, ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        return [self.paragraph(ext_word_list=ext_word_list) for _ in range(0, nb)]

    def text(self, max_nb_chars: int = 200, ext_word_list: Optional[Sequence[str]] = None) -> str:
        """Generate a text string.

        The ``max_nb_chars`` argument is an upper bound, not a target: the
        result will never be longer than ``max_nb_chars``, but it can be
        substantially shorter. This is because the text is built from whole
        words, sentences, or paragraphs (depending on the value of
        ``max_nb_chars``), and the last complete unit that would push the
        length past the limit is dropped. For example, ``max_nb_chars=200``
        regularly produces strings of 100-190 characters. Values below 5
        raise a ``ValueError``. The ``ext_word_list`` argument works in
        exactly the same way it would in |words|, |sentences|, or
        |paragraphs|.

        :sample: max_nb_chars=20
        :sample: max_nb_chars=80
        :sample: max_nb_chars=160
        :sample: ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        text: List[str] = []
        if max_nb_chars < 5:
            raise ValueError("text() can only generate text of at least 5 characters")

        if max_nb_chars < 25:
            # join words
            while not text:
                size = 0
                # determine how many words are needed to reach the $max_nb_chars
                # once;
                while size < max_nb_chars:
                    word = (self.word_connector if size else "") + self.word(ext_word_list=ext_word_list)
                    text.append(word)
                    size += len(word)
                text.pop()
            text[0] = text[0][0].upper() + text[0][1:]
            last_index = len(text) - 1
            text[last_index] += self.sentence_punctuation
        elif max_nb_chars < 100:
            # join sentences
            while not text:
                size = 0
                # determine how many sentences are needed to reach the
                # $max_nb_chars once
                while size < max_nb_chars:
                    sentence = (self.word_connector if size else "") + self.sentence(ext_word_list=ext_word_list)
                    text.append(sentence)
                    size += len(sentence)
                text.pop()
        else:
            # join paragraphs
            while not text:
                size = 0
                # determine how many paragraphs are needed to reach the
                # $max_nb_chars once
                while size < max_nb_chars:
                    paragraph = ("\n" if size else "") + self.paragraph(ext_word_list=ext_word_list)
                    text.append(paragraph)
                    size += len(paragraph)
                text.pop()

        return "".join(text)

    def texts(
        self, nb_texts: int = 3, max_nb_chars: int = 200, ext_word_list: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Generate a list of text strings.

        The ``nb_texts`` argument controls how many text strings the list will
        contain, and this method uses |text| under the hood for text generation,
        so the two remaining arguments, ``max_nb_chars`` and ``ext_word_list``
        will work in exactly the same way as well.

        :sample: nb_texts=5
        :sample: nb_texts=5, max_nb_chars=50
        :sample: nb_texts=5, max_nb_chars=50,
                 ext_word_list=['abc', 'def', 'ghi', 'jkl']
        """
        return [self.text(max_nb_chars, ext_word_list) for _ in range(0, nb_texts)]

    def text_with_exact_byte_length(
        self,
        fragments: Sequence[str],
        separator: str = " ",
        *,
        nb_bytes: int,
        unicode_form: str = "NFC",
        max_operations: int = 1_000_000,
    ) -> str:
        """Generate a string whose normalized UTF-8 encoding has an exact size.

        Unlike |text|, whose ``max_nb_chars`` is an upper bound measured in
        characters, this method targets an exact number of UTF-8 bytes
        *after* Unicode normalization. The result is a concatenation of
        fragments from ``fragments`` (fragments may repeat) joined by
        ``separator``. No fragment is ever truncated or split, and the size
        is never padded with extra spaces: if the target cannot be hit with
        the supplied pieces, an exception is raised instead.

        ``fragments`` must be a non-empty sequence. Fragments and the
        separator are normalized with ``unicode_form`` (one of ``NFC``,
        ``NFD``, ``NFKC``, ``NFKD``); fragments that are equal after
        normalization are treated as the same candidate, and a fragment that
        becomes empty after normalization is rejected, since it would create
        a zero-length loop. The separator must likewise be non-empty *after*
        normalization: an empty separator scheme would allow joins to grow
        the piece count without consuming any bytes. ``nb_bytes`` must be
        positive.

        Bytes are counted on the normalized whole string, which matters when
        combining characters meet across a join boundary, e.g. a fragment
        ending in a base letter followed by a separator or fragment starting
        with a combining mark: NFC composes the pair into one code point.

        If no combination of the fragments can ever total ``nb_bytes``,
        |UnsatisfiableByteLength| is raised. If the exhaustive search needs
        more than ``max_operations`` normalizations to decide,
        |ByteLengthBudgetExhausted| is raised instead; that means the answer
        is unknown, not that no solution exists.

        :sample: fragments=['hello', '你好', '世界'], nb_bytes=12
        :sample: fragments=['b', 'e'], separator='\\u0300', nb_bytes=3
        """
        if isinstance(nb_bytes, bool) or not isinstance(nb_bytes, int):
            raise TypeError("nb_bytes must be an integer")
        if isinstance(max_operations, bool) or not isinstance(max_operations, int):
            raise TypeError("max_operations must be an integer")
        if nb_bytes <= 0:
            raise ValueError("nb_bytes must be positive")
        if max_operations <= 0:
            raise ValueError("max_operations must be positive")
        if unicode_form not in _NORMALIZATION_FORMS:
            raise ValueError(f"unicode_form must be one of {', '.join(_NORMALIZATION_FORMS)}")
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        if not fragments:
            raise ValueError("fragments must be a non-empty sequence")

        remaining_operations = max_operations

        def normalize(value: str) -> str:
            nonlocal remaining_operations
            remaining_operations -= 1
            if remaining_operations < 0:
                raise ByteLengthBudgetExhausted(
                    f"used more than {max_operations} normalization operations while "
                    f"searching for an exact {nb_bytes}-byte string; the answer is unknown"
                )
            return unicodedata.normalize(unicode_form, value)

        normalized_separator = normalize(separator)
        if not normalized_separator:
            raise ValueError(
                "separator is empty after normalization; an empty separator scheme "
                "would allow zero-length joins"
            )

        # Fragments equal after normalization are one and the same value.
        # The first original spelling is kept so the returned string is built
        # from caller-provided pieces verbatim.
        originals: Dict[str, str] = {}
        for fragment in fragments:
            if not isinstance(fragment, str):
                raise TypeError("fragments must contain only strings")
            normalized_fragment = normalize(fragment)
            if not normalized_fragment:
                raise ValueError(
                    f"fragment {fragment!r} is empty after normalization; empty "
                    "fragments would create zero-length loops"
                )
            originals.setdefault(normalized_fragment, fragment)
        normalized_fragments: Tuple[str, ...] = tuple(originals)

        # A state is (interactive tail, normalized size in bytes). The tail
        # is the suffix starting at the last combining-class-0 character:
        # normalization of an appended string can only interact with that
        # suffix. The size belongs to the state too, since the same tail can
        # occur at several sizes. A transition appends the separator (except
        # at the root) and one fragment, then normalizes. Weights are byte
        # deltas, always non-negative, so states are enumerated by size with
        # zero-weight edges closed first through a bounded queue, which keeps
        # zero-weight cycles finite. parent[state] records the previous state
        # and the original spelling of the fragment used for the transition.
        State = Tuple[str, int]
        root: State = ("", 0)
        parent: Dict[State, Tuple[State, str]] = {}
        levels: Dict[int, Dict[State, None]] = {0: {root: None}}

        size = 0
        while size < nb_bytes:
            current = levels.get(size)
            if current is None:
                size += 1
                continue

            queue: Deque[State] = deque(current)
            while queue:
                tail, state_size = queue.popleft()
                prefix_part = tail + normalized_separator if tail else ""
                for normalized_fragment in normalized_fragments:
                    next_text = normalize(prefix_part + normalized_fragment)
                    next_size = state_size + len(next_text.encode("utf-8")) - len(tail.encode("utf-8"))
                    next_state = (_interactive_tail(next_text), next_size)
                    if next_size == state_size and next_state not in parent:
                        parent[next_state] = ((tail, state_size), originals[normalized_fragment])
                        current[next_state] = None
                        queue.append(next_state)

            for tail, state_size in tuple(current):
                prefix_part = tail + normalized_separator if tail else ""
                for normalized_fragment in normalized_fragments:
                    next_text = normalize(prefix_part + normalized_fragment)
                    next_size = state_size + len(next_text.encode("utf-8")) - len(tail.encode("utf-8"))
                    if next_size <= state_size or next_size > nb_bytes:
                        continue
                    next_state = (_interactive_tail(next_text), next_size)
                    if next_state not in parent:
                        parent[next_state] = ((tail, state_size), originals[normalized_fragment])
                    levels.setdefault(next_size, {})[next_state] = None

            size += 1

        target_state = next(iter(levels.get(nb_bytes, ())), None)
        if target_state is None:
            raise UnsatisfiableByteLength(
                f"no combination of the {len(normalized_fragments)} fragment(s) "
                f"separated by {separator!r} can produce exactly {nb_bytes} "
                f"bytes in {unicode_form}"
            )

        pieces: List[str] = []
        node: State = target_state
        while node != root:
            previous, fragment = parent[node]
            pieces.append(fragment)
            node = previous
        pieces.reverse()
        result = separator.join(pieces)
        assert len(unicodedata.normalize(unicode_form, result).encode("utf-8")) == nb_bytes
        return result
