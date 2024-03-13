import pickle

import re
import numpy as np

from typing import List, Iterable, Optional, Dict
from collections import defaultdict

class Tokenizer:
    def __init__(self,
                 token_pattern: str = '\w+|[\!\?\,\.\-\:]',
                 eos_token: str = '<EOS>',
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>'):
        self.token_pattern = token_pattern
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.special_tokens = [self.eos_token, self.pad_token, self.unk_token]
        self.vocab = None
        self.inverse_vocab = None

    def text_preprocess(self, input_text: str) -> str:
        """ Предобрабатываем один текст """
        input_text = input_text.lower()  # приведение к нижнему регистру
        input_text = re.sub('\s+', ' ', str(input_text))  # унифицируем пробелы
        input_text = input_text.strip()
        return input_text

    def build_vocab(self, corpus: List[str]) -> None:
        assert len(corpus)
        all_tokens = set()
        for text in corpus:
            all_tokens |= set(self._tokenize(text, append_eos_token=False))
        self.vocab = {elem: ind for ind, elem in enumerate(all_tokens)}
        special_tokens = [self.eos_token, self.unk_token, self.pad_token]
        for token in special_tokens:
            self.vocab[token] = len(self.vocab) + 1
        self.inverse_vocab = {ind: elem for elem, ind in self.vocab.items()}
        return self

    def _tokenize(self, text: str, append_eos_token: bool = True) -> List[str]:
        text = self.text_preprocess(text)
        tokens = re.findall(self.token_pattern, text)
        if append_eos_token:
            tokens.append(self.eos_token)
        return tokens

    def encode(self, text: str, append_eos_token: bool = False) -> List[str]:
        """ Токенизируем текст """
        tokens = self._tokenize(text, append_eos_token)
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return ids

    def decode(self, input_ids: Iterable[int], remove_special_tokens: bool = False) -> str:
        assert len(input_ids)
        assert max(input_ids) < len(self.vocab) and min(input_ids) >= 0
        tokens = []
        for ind in input_ids:
            token = self.inverse_vocab[ind]
            if remove_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        text = ' '.join(tokens)
        return text

class StatLM:
    def __init__(self,
                 # vocab: Dict[str, int],
                 tokenizer: Tokenizer,
                 context_size: int = 2,
                 alpha: float = 0.1,
                 sample_top_p: Optional[float] = None
                 ):

        assert context_size >= 2
        assert sample_top_p is None or 0.0 < sample_top_p

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.sample_top_p = sample_top_p

        self.n_gramms_stat = defaultdict(int)
        self.nx_gramms_stat = defaultdict(int)

    def get_token_by_ind(self, ind: int) -> str:
        return self.tokenizer.vocab.get(ind)

    def get_ind_by_token(self, token: str) -> int:
        return self.tokenizer.inverse_vocab.get(token, self.tokenizer.inverse_vocab[self.unk_token])

    # def train(self, train_token_indices: List[List[int]]):
    def train(self, train_texts: List[str]):
        for sentence in train_texts:
            sentence_ind = self.tokenizer.encode(sentence)
            for i in range(len(sentence_ind) - self.context_size):
                seq = tuple(sentence_ind[i: i + self.context_size - 1])
                self.n_gramms_stat[seq] += 1

                seq_x = tuple(sentence_ind[i: i + self.context_size])
                self.nx_gramms_stat[seq_x] += 1

            seq = tuple(sentence_ind[len(sentence_ind) - self.context_size:])
            self.n_gramms_stat[seq] += 1

    def sample_token(self, token_distribution: np.ndarray) -> int:
        if self.sample_top_p is None:
            return token_distribution.argmax()
        else:
            token_distribution = sorted(list(zip(token_distribution, np.arange(len(token_distribution)))))
            total_proba = 0.0
            tokens_to_sample = []
            tokens_probas = []
            if self.sample_top_p < 1:
                for token_proba, ind in sorted(token_distribution, reverse=True):
                    tokens_to_sample.append(ind)
                    tokens_probas.append(token_proba)
                    total_proba += token_proba
                    if total_proba >= self.sample_top_p:
                        break
            else:
                counter = 0
                for token_proba, ind in sorted(token_distribution, reverse=True):
                    tokens_to_sample.append(ind)
                    tokens_probas.append(token_proba)
                    counter += 1
                    if counter >= self.sample_top_p:
                        break
            # для простоты отнормируем вероятности, чтобы суммировались в единицу
            tokens_probas = np.array(tokens_probas)
            tokens_probas = tokens_probas / tokens_probas.sum()
            return np.random.choice(tokens_to_sample, p=tokens_probas)

    def get_stat(self) -> Dict[str, Dict]:

        n_token_stat, nx_token_stat = {}, {}
        for token_inds, count in self.n_gramms_stat.items():
            n_token_stat[self.tokenizer.decode(token_inds)] = count

        for token_inds, count in self.nx_gramms_stat.items():
            nx_token_stat[self.tokenizer.decode(token_inds)] = count

        return {
            'n gramms stat': self.n_gramms_stat,
            'n+1 gramms stat': self.nx_gramms_stat,
            'n tokens stat': n_token_stat,
            'n+1 tokens stat': nx_token_stat,
        }

    def _get_next_token(self, tokens: List[int]) -> (int, str):
        denominator = self.n_gramms_stat.get(tuple(tokens), 0) + self.alpha * len(self.tokenizer.vocab)
        numerators = []
        for ind in self.tokenizer.inverse_vocab:
            numerators.append(self.nx_gramms_stat.get(tuple(tokens + [ind]), 0) + self.alpha)

        token_distribution = np.array(numerators) / denominator
        while True:
            max_proba_ind = self.sample_token(token_distribution)
            try:
                next_token = self.tokenizer.inverse_vocab[max_proba_ind]
                break
            except Exception as e:
                print(f'Error: {e}')
        return max_proba_ind, next_token

    def generate_token(self, text: str, remove_special_tokens: bool = False) -> Dict:
        tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = tokens[-self.context_size + 1:]

        max_proba_ind, next_token = self._get_next_token(tokens)

        return {
            'next_token': next_token,
            'next_token_num': max_proba_ind,
        }

    def generate_text(self, text: str, max_tokens: int, remove_special_tokens: bool = False) -> Dict:
        all_tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = all_tokens[-self.context_size + 1:]

        next_token = None
        while next_token != self.tokenizer.eos_token and len(all_tokens) < max_tokens:
            max_proba_ind, next_token = self._get_next_token(tokens)
            all_tokens.append(max_proba_ind)
            tokens = all_tokens[-self.context_size + 1:]

        new_text = self.tokenizer.decode(all_tokens, remove_special_tokens)

        finish_reason = 'max tokens'
        if all_tokens[-1] == self.tokenizer.vocab[self.tokenizer.eos_token]:
            finish_reason = 'end of text'

        return {
            'all_tokens': all_tokens,
            'total_text': new_text,
            'finish_reason': finish_reason
        }


def construct_model():
    with open("models/stat_llm.pkl", "rb") as f:
        model = pickle.load(f)
        return model, {'max_tokens': 16}
