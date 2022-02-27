import json
from pathlib import Path

import torch

from constants import ENCODING

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"


class CharTokenizer:
    def __init__(self, vocab):
        self._char_to_idx = vocab
        self._idx_to_char = {v: k for k, v in self._char_to_idx.items()}
        self._vocab_size = len(self._char_to_idx)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def pad_token(self):
        return self._char_to_idx[PAD]

    def char_to_idx_get(self, x):
        return self._char_to_idx.get(x, self._char_to_idx[UNK])

    def idx_to_char_get(self, x):
        return self._idx_to_char[x]

    @classmethod
    def from_disk(cls, path):
        with open(path, "r", encoding=ENCODING) as f:
            return CharTokenizer(json.load(f))

    def to_disk(self, path: str) -> None:
        path = Path(path)
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        with open(path, "w", encoding=ENCODING) as f:
            json.dump(self._char_to_idx, f, ensure_ascii=True, indent=2)


class CharLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.hid_to_logits = torch.nn.Linear(in_features=self.hidden_size,
                                             out_features=self.input_size)

    def forward(self, x, hid_state):
        x = torch.nn.functional.one_hot(x, num_classes=self.input_size).float()
        h_seq, (h_0, c_0) = self.rnn(x, hid_state)
        next_logits = self.hid_to_logits(h_seq)
        next_logp = torch.nn.functional.log_softmax(next_logits, dim=-1)
        return next_logp, (h_0, c_0)

    def initial_state(self, batch_size):
        """ return rnn state before it processes first input (aka h0) """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False),  # pylint: disable=no-member
            torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)  # pylint: disable=no-member
        )

    def to_disk(self, path):
        path = Path(path)
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, path)

    @classmethod
    def from_disk(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"]
        )
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        return model
