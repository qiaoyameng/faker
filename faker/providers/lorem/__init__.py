from collections import deque
from typing import Collection, Deque, Dict, List, Optional, Sequence, Set, Tuple, cast
from unicodedata import normalize

from ...exceptions import ImpossibleTextLength, TextGenerationBudgetExceeded
from .. import BaseProvider

localized = True

# 'Latin' is the default locale
default_locale = "la"


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

    @staticmethod
    def _normalize_text(value: str, normalization: str) -> str:
        try:
            return normalize(normalization, value)
        except (LookupError, TypeError) as exc:
            raise ValueError(f"Unsupported Unicode normalization form: {normalization}") from exc

    @staticmethod
    def _normalized_pieces(
        fragments: Collection[str],
        separator: str,
        normalization: str,
    ) -> Tuple[List[str], str]:
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")

        normalized_separator = Provider._normalize_text(separator, normalization)
        if not normalized_separator:
            raise ValueError("separator must contain at least one character after normalization")

        normalized_fragments_by_value: Dict[str, str] = {}
        for fragment in fragments:
            if not isinstance(fragment, str):
                raise TypeError("fragments must be strings")

            normalized_fragment = Provider._normalize_text(fragment, normalization)
            if not normalized_fragment:
                raise ValueError("fragments must contain at least one character after normalization")

            normalized_fragments_by_value.setdefault(normalized_fragment, normalized_fragment)

        if not normalized_fragments_by_value:
            raise ValueError("at least one fragment is required")

        return list(normalized_fragments_by_value), normalized_separator

    def _find_exact_byte_text(
        self,
        fragments: List[str],
        separator: str,
        target_nb_bytes: int,
        normalization: str,
        budget: int,
    ) -> str:
        queue: Deque[str] = deque()
        seen: Set[str] = set()
        attempts = 0

        queue.append("")
        seen.add("")

        while queue:
            current = queue.popleft()
            suffixes = list(fragments) if not current else [separator + fragment for fragment in fragments]
            self.generator.random.shuffle(suffixes)

            for suffix in suffixes:
                attempts += 1
                if attempts > budget:
                    raise TextGenerationBudgetExceeded(
                        f"Search budget of {budget} transitions was exhausted before an exact "
                        f"{target_nb_bytes}-byte text was found or proven impossible."
                    )

                candidate = self._normalize_text(current + suffix, normalization)
                candidate_bytes = len(candidate.encode("utf-8"))

                if candidate_bytes > target_nb_bytes or candidate in seen:
                    continue
                if candidate_bytes == target_nb_bytes:
                    return candidate

                seen.add(candidate)
                queue.append(candidate)

        raise ImpossibleTextLength(
            f"No combination of the supplied fragments has exactly {target_nb_bytes} UTF-8 bytes "
            f"after {normalization} normalization."
        )

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

    def text_bytes(
        self,
        fragments: Optional[Collection[str]] = None,
        target_nb_bytes: int = 16,
        normalization: str = "NFC",
        separator: str = " ",
        budget: int = 10000,
    ) -> str:
        """Generate text with an exact normalized UTF-8 byte length.

        When ``fragments`` is ``None``, a small built-in set of fragments is
        used.

        Fragments may be repeated. Fragments that are equal after
        normalization are treated as a single candidate. Fragments and
        separators are never truncated, and padding is not added.

        :sample: fragments=['hello', 'world'], target_nb_bytes=11
        :sample: fragments=['a', '中'], target_nb_bytes=5
        :sample: fragments=['e', 'x'], target_nb_bytes=5, separator='\u0301'
        """
        if type(target_nb_bytes) is not int:
            raise TypeError("target_nb_bytes must be an integer")
        if type(budget) is not int:
            raise TypeError("budget must be an integer")
        if target_nb_bytes <= 0:
            raise ValueError("target_nb_bytes must be greater than 0")
        if budget <= 0:
            raise ValueError("budget must be greater than 0")
        if fragments is None:
            fragments = ["lorem", "ipsum", "dolor", "sit", "amet"]

        normalized_fragments, normalized_separator = self._normalized_pieces(
            fragments,
            separator,
            normalization,
        )

        return self._find_exact_byte_text(
            normalized_fragments,
            normalized_separator,
            target_nb_bytes,
            normalization,
            budget,
        )

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
