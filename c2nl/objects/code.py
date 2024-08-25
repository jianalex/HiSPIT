from c2nl.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD

import numpy as np
import torch

class Code(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._language = None
        self._text = None
        self._tokens = []
        self._type = []
        self._mask = []
        self.src_vocab = None  # required for Copy Attention
        # for code structure
        self._struc = None       #for stmt adjacency matrix
        self._struc_code = None  #for pdg adjacency matrix

    @property
    def id(self) -> str:
        return self._id

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, param: str) -> None:
        self._language = param

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def type(self) -> list:
        return self._type

    @type.setter
    def type(self, param: list) -> None:
        assert isinstance(param, list)
        self._type = param

    @property
    def mask(self) -> list:
        return self._mask

    @mask.setter
    def mask(self, param: list) -> None:
        assert isinstance(param, list)
        self._mask = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()

    # for code structure
    @property
    def struc(self) -> np.ndarray:
        return self._struc

    @struc.setter
    def struc(self, param: np.ndarray) -> None:
        assert isinstance(param, np.ndarray)
        self._struc = param
    
    @property
    def struc_code(self) -> np.ndarray:
        return self._struc_code

    @struc_code.setter
    def struc_code(self, param: np.ndarray) -> None:
        assert isinstance(param, np.ndarray)
        self._struc_code = param
        
    #----------------------------------------------------------------
    def form_src_vocab(self) -> None:
        self.src_vocab = Vocabulary()
        assert self.src_vocab.remove(BOS_WORD)
        assert self.src_vocab.remove(EOS_WORD)
        self.src_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False
