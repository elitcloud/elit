# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-04 14:32
class ConllWord(object):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, relation=None, phead=None,
                 pdeprel=None):
        """CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : str
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        cpos : str
            Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        pos : str
            Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        feats : str
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or an underscore if not available.
        head : int
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        relation : str
            Dependency relation to the HEAD.
        phead : int
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        pdeprel : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.relation = relation
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  None if self.head is None else str(self.head), self.relation, self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])


class ConllSentence(object):
    def __init__(self, words):
        """A list of ConllWord

        Parameters
        ----------
        words : Sequence[ConllWord]
            words of a sentence
        """
        super().__init__()
        self.words = words

    def __str__(self):
        return '\n'.join([word.__str__() for word in self.words])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.words[index]

    def __iter__(self):
        return (line for line in self.words)
