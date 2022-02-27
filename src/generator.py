import click
import torch
import numpy as np

from model import BOS, EOS, CharTokenizer, CharLSTM
from constants import MAX_LEN

STOP_LIST = [EOS, " "] + list('!"#$%&()*+,-./:;<=>?@[\\]^_{|}~')
CUDA = "cuda"
CPU = "cpu"


@click.group()
def main():
    pass


def to_matrix(names, tokenizer, max_len=None, dtype='int32', batch_first=True):
    """Casts a list of names into rnn-digestable matrix"""
    max_len = max_len or max(map(len, names))
    names_ix = np.zeros([len(names), max_len], dtype) + tokenizer.pad_token

    for i in range(len(names)):
        line_ix = [tokenizer.char_to_idx_get(c) for c in names[i]]
        names_ix[i, :len(line_ix)] = line_ix

    if not batch_first:  # convert [batch, time] into [time, batch]
        names_ix = np.transpose(names_ix)

    return names_ix


def generate_sample_lstm(model, tokenizer, max_length, seed_phrase, temperature, device):
    with torch.no_grad():
        answer = [BOS] + list(seed_phrase)

        x_sequence = torch.tensor(to_matrix([answer], tokenizer), dtype=torch.long)  # pylint: disable=no-member,not-callable
        x_sequence = x_sequence.to(device)
        h_0, c_0 = model.initial_state(1)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        logp_seq, (h_0, c_0) = model(x_sequence, (h_0, c_0))
        logp_seq = logp_seq[:, -1, :]

        for _ in range(max_length - len(seed_phrase)):
            p_next = torch.nn.functional.softmax(logp_seq.data.cpu() / temperature, dim=-1)
            p_next = p_next.data.numpy()[0]

            next_ix = np.random.choice(tokenizer.vocab_size, p=p_next)
            next_ix = tokenizer.idx_to_char_get(next_ix)

            answer.append(next_ix)

            if next_ix == EOS:
                break

            x_sequence = torch.tensor(to_matrix([[next_ix]], tokenizer), dtype=torch.long)  # pylint: disable=no-member,not-callable
            x_sequence = x_sequence.to(device)
            logp_seq, (h_0, c_0) = model(x_sequence, (h_0, c_0))
            logp_seq = logp_seq[:, -1, :]
    return ''.join(answer[1:-1])


def autocomplete_beam_search(model, tokenizer, max_length, seed_phrase, beam_size, stop_list,
                             device):
    with torch.no_grad():
        seed_phrase = [BOS] + list(seed_phrase)
        candidates = [(seed_phrase, 0, len(seed_phrase))]
        is_start = True

        for _ in range(max_length - len(seed_phrase)):
            new_candidates = []
            for trg_indexes, log_prob_sum, cnt in candidates:
                if is_start or (trg_indexes[-1] not in stop_list):
                    x_sequence = torch.tensor(to_matrix([trg_indexes], tokenizer), dtype=torch.long)  # pylint: disable=no-member,not-callable
                    x_sequence = x_sequence.to(device)

                    h_0, c_0 = model.initial_state(1)
                    h_0 = h_0.to(device)
                    c_0 = c_0.to(device)

                    logp_seq, (h_0, c_0) = model(x_sequence, (h_0, c_0))
                    logp_seq = logp_seq[:, -1, :]

                    topvs, topis = logp_seq.data.cpu().view(-1).topk(beam_size)

                    for topv, topi in zip(topvs, topis):
                        next_ix = trg_indexes + [tokenizer.idx_to_char_get(topi.item())]
                        new_cnt = cnt + 1
                        new_log_prob_sum = log_prob_sum + topv.item()
                        new_candidates.append((next_ix, new_log_prob_sum, new_cnt))
                else:
                    new_candidates.append((trg_indexes, log_prob_sum, cnt))
            is_start = False
            new_candidates = sorted(
                new_candidates,
                key=lambda x: x[1] / x[2],
                reverse=True
            )
            candidates = new_candidates[:beam_size]

    return [
        "".join(candidates[0][1:]) if candidates[0][-1] != EOS else "".join(candidates[0][1:-1])
        for candidates in candidates
    ]


@main.command()
@click.option("--model-path", required=True, type=str)
@click.option("--tokenizer-path", required=True, type=str)
@click.option("--seed-phrase", required=True, type=str)
def generate(model_path, tokenizer_path, seed_phrase):
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)  # pylint: disable=no-member

    model = CharLSTM.from_disk(model_path, device)
    tokenizer = CharTokenizer.from_disk(tokenizer_path)

    for t in [0.1, 0.2, 0.5, 1.0]:
        print(f'===={t}====')
        result = generate_sample_lstm(model=model, tokenizer=tokenizer, max_length=MAX_LEN,
                                      seed_phrase=seed_phrase, temperature=t, device=device)
        print(result)
        print('==========\n')


@main.command()
@click.option("--model-path", required=True, type=str)
@click.option("--tokenizer-path", required=True, type=str)
@click.option("--seed-phrase", required=True, type=str)
@click.option("--beam-size", required=True, type=int)
def autocompletion(model_path, tokenizer_path, seed_phrase, beam_size):
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)  # pylint: disable=no-member

    model = CharLSTM.from_disk(model_path, device)
    tokenizer = CharTokenizer.from_disk(tokenizer_path)

    result = autocomplete_beam_search(model, tokenizer, max_length=MAX_LEN,
                                      seed_phrase=seed_phrase, beam_size=beam_size,
                                      stop_list=STOP_LIST, device=device)
    print(result)


if __name__ == "__main__":
    main()
