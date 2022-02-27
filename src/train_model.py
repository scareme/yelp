import time
import random
import warnings

import click
import torch
from torchtext.data import Field, TabularDataset, BucketIterator

from model import BOS, EOS, PAD, UNK, CharTokenizer, CharLSTM
from constants import TEXT

CUDA = "cuda"
CPU = "cpu"
SEED = 97
IGNORE_WARNINGS = True
# TRAIN_SIZE = 0.8
# BATCH_SIZE = 256
# HIDDEN_SIZE = 256
# NUM_LAYERS = 2
# LR = 0.001
# CLIP = 0.1
# N_EPOCHS = 20

TRAIN_SIZE = 0.01
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LR = 0.001
CLIP = 0.1
N_EPOCHS = 1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_train_test_vocab(path, device):
    text_field = Field(tokenize=lambda x: x.lower(),
                       init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                       batch_first=True)
    data = TabularDataset(path, format="JSON", fields={TEXT: (TEXT, text_field)})
    text_field.build_vocab(data)
    train_data, test_data = data.split([TRAIN_SIZE, 1 - TRAIN_SIZE], random_state=random.getstate())

    train_iterator = BucketIterator(train_data, train=True, batch_size=BATCH_SIZE, shuffle=True,
                                    sort=False, sort_within_batch=False, device=device)
    test_iterator = BucketIterator(test_data, train=False, batch_size=BATCH_SIZE, shuffle=False,
                                   sort=False, sort_within_batch=False, device=device)

    vocab = dict(text_field.vocab.stoi)
    return train_iterator, test_iterator, vocab


def train(model, iterator, tokenizer, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()

        batch = batch.text
        h_0, c_0 = model.initial_state(batch.shape[0])
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        logp_seq, _ = model(batch, (h_0, c_0))

        output = logp_seq[:, :-1].contiguous().view(-1, tokenizer.vocab_size)
        trg = batch[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.detach().item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, tokenizer, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            batch = batch.text
            h_0, c_0 = model.initial_state(batch.shape[0])
            h_0 = h_0.to(device)
            c_0 = c_0.to(device)

            logp_seq, _ = model(batch, (h_0, c_0))

            output = logp_seq[:, :-1].contiguous().view(-1, tokenizer.vocab_size)
            trg = batch[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.detach().item()
    return epoch_loss / len(iterator)


def train_n_epochs(n_epochs, model, train_iterator, test_iterator, tokenizer, optimizer, criterion,
                   clip, device, path_to_save_model):
    best_valid_loss = float('inf')
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        train_loss = train(model=model, iterator=train_iterator, tokenizer=tokenizer,
                           optimizer=optimizer, criterion=criterion, clip=clip, device=device)
        valid_loss = evaluate(model=model, iterator=test_iterator, tokenizer=tokenizer,
                              criterion=criterion, device=device)

        epoch_end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start_time, epoch_end_time)
        total_mins, total_secs = epoch_time(start_time, epoch_end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.to_disk(path_to_save_model)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
              f" | Total time: {total_mins}m {total_secs}s")
        print(f'\tTrain Loss: {train_loss:.5f}')
        print(f'\t Val. Loss: {valid_loss:.5f} | Best Loss: {best_valid_loss:.5f}')


@click.command()
@click.option("--reviews-path", required=True, type=str)
@click.option("--path-to-save-model", required=True, type=str)
@click.option("--path-to-save-tokenizer", required=True, type=str)
def main(reviews_path, path_to_save_model, path_to_save_tokenizer):
    random.seed(SEED)
    if IGNORE_WARNINGS:
        warnings.filterwarnings("ignore")
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)  # pylint: disable=no-member

    train_iterator, test_iterator, vocab = get_train_test_vocab(path=reviews_path, device=device)

    tokenizer = CharTokenizer(vocab)
    tokenizer.to_disk(path_to_save_tokenizer)
    print(f"Tokenizer saved at {path_to_save_tokenizer}")

    model_lstm = CharLSTM(input_size=tokenizer.vocab_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model_lstm = model_lstm.to(device)
    optimizer = torch.optim.AdamW(model_lstm.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss(ignore_index=tokenizer.pad_token)

    print(f"Training model with n_epochs = {N_EPOCHS}, at device = {device}:")
    train_n_epochs(n_epochs=N_EPOCHS,
                   model=model_lstm,
                   train_iterator=train_iterator,
                   test_iterator=test_iterator,
                   tokenizer=tokenizer,
                   optimizer=optimizer,
                   criterion=criterion,
                   clip=CLIP,
                   device=device,
                   path_to_save_model=path_to_save_model)
    print(f"The best model was saved at {path_to_save_model}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
