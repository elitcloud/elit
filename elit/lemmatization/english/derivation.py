from typing import Sequence
from elit.lemmatization.english.suffix_group import SuffixGroup
class Derivation:
    def __init__(self,suffix_groups: Sequence[SuffixGroup]):
        self.suffix_groups = suffix_groups
        assert(suffix_groups != None), "suffix rule can not be empty."

    def getSuffixMatchers(self)->Sequence[SuffixGroup]:
        return self.suffix_groups


    def get_base_form(self, lower: str, pos: str):
        """
        Enumerate each suffix group.
        :param lower:
        :param pos:
        :return:
        """
        for suffix_group in self.suffix_groups:
            base, suffix, suffix_tag= suffix_group.get_base_form_tag(lower, pos)
            if base is not None:
                return base, suffix, suffix_tag

        return None,None, None