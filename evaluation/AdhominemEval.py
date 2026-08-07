import logging
from collections import OrderedDict
import json
import torch
from torch.utils.data import DataLoader

from valla.methods.torched_AdHominem import (
    AdHominem,
    AVDataset as BaseAVDataset,
    load_model_from_disk,
    evaluate_model,
    modified_contrastive_loss,
data_to_device, kernel_fn, euclidean_dist
)

# build vocab function
def build_vocab(train_path, tok_file, chr_file, tok_count_min, chr_count_min, tok_vocab_size, chr_vocab_size):
    tok_file = tok_file if tok_file is not None else f'{train_path}.adhom.tok_count'
    logging.info(f'Getting token count file from {tok_file}')
    with open(tok_file, 'r') as tok_counts_file:
        tok_counts = json.load(tok_counts_file)

    chr_file = chr_file if chr_file is not None else f'{train_path}.adhom.chr_count'
    logging.info(f'Getting character count file from {chr_file}')
    with open(chr_file, 'r') as char_counts_file:
        char_counts = json.load(char_counts_file)

    char_vocab = [[char, int(char_count)] for char, char_count in char_counts.items() if int(char_count) >= chr_count_min]
    char_vocab = OrderedDict([(x[0], x[1]) for x in sorted(char_vocab, key=lambda x: x[1], reverse=True)[:chr_vocab_size]])

    tok_vocab = [[tok, int(tok_count)] for tok, tok_count in tok_counts.items() if int(tok_count) >= tok_count_min]
    tok_vocab = OrderedDict([(x[0], x[1]) for x in sorted(tok_vocab, key=lambda x: x[1], reverse=True)[:tok_vocab_size]])

    return tok_vocab, char_vocab

# dataset class for input lists
class AVDataset(BaseAVDataset):
    def __init__(self, input_list, *args, **kwargs):
        super(AVDataset, self).__init__(*args, **kwargs)
        if input_list is None or len(input_list) == 0:
            raise ValueError("input_list must be provided and non-empty")
        logging.info(f'Loading dataset from input list with {len(input_list)} samples')
        self.data = input_list
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        label, doc0, doc1 = self.data[item]

        if isinstance(doc0, dict):
            doc0 = doc0.get('text', doc0)
        if isinstance(doc1, dict):
            doc1 = doc1.get('text', doc1)

        doc0 = self.preprocess_doc(doc0[:self.max_txt_len])
        doc1 = self.preprocess_doc(doc1[:self.max_txt_len])

        chars0, tokens0 = doc0['chars'], doc0['tokens']
        chars1, tokens1 = doc1['chars'], doc1['tokens']

        label = 1 if label == 1 else -1

        chars0, tokens0, ws_do_mask0, sd_do_mask0, ws_lens0, sd_lens0 = self.ids_to_tokens(chars0, tokens0)
        chars1, tokens1, ws_do_mask1, sd_do_mask1, ws_lens1, sd_lens1 = self.ids_to_tokens(chars1, tokens1)

        return torch.tensor(label), tokens0, tokens1, chars0, chars1, \
               ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, \
               ws_lens0, ws_lens1, sd_lens0, sd_lens1


class ADHOMINEM:
    def __init__(self, model_type):
        # add paths to model.torch adhominem files here
        self.model_paths = {
            "twitter": "",
            "cross": "",
            "in": "",
            "profile": "",
            "mixed": ""
        }

        # add paths to train.jsonl files here
        self.train_paths = {
            "twitter": "/data/twitter/train.jsonl",
            "cross": "/data/cross_domain/train.jsonl",
            "in": "/data/in_domain/train.jsonl",
            "profile": "/data/profile_based/train.jsonl",
            "mixed": "/data/mixed/train.jsonl"
        }


        self.model_path = self.model_paths[model_type]
        self.train_path = self.train_paths[model_type]

        # args as in training
        class Args:
            device = 0
            logging_steps = 2000
            save_best_model = True
            save_model_checkpoints = True
            save_model_dir = 'adhominem_torched'
            train_path = self.train_path
            test_path = None
            num_dataloader_workers = 10
            epochs = 10
            model_path = self.model_path
            train_batch_size = 4
            lr = 0.0002
            weight_decay = 0
            loss_margin = 0.05
            cnn_stride = 1
            D_c = 10
            D_r = 30
            w = 4
            D_w = 300
            D_s = 50
            D_d = 50
            D_mlp = 60
            max_chars_per_word = 15
            max_words_per_sentence = 30
            max_sentences_per_doc = 50
            cnn_dropout_prob = 0.2
            w2s_dropout_prob = 0.1
            w2s_att_dropout_prob = 0.1
            s2d_dropout_prob = 0.1
            s2d_att_dropout_prob = 0.1
            metric_dropout_prob = 0.2
            chr_vocab_size = 250
            tok_vocab_size = 5000
            dont_use_fasttext = True
            max_grad_norm = 1
            lr_decay_gamma = 0.96
            chr_count_min = 100
            tok_count_min = 10
            tok_file = None
            chr_file = None
            evaluate_every_epoch = False
            evaluation_steps = 2000
            test_batch_size = 4
            AA = False

        self.args = Args()

    def __call__(self, i1, i2, threshold=0.5):
        if not isinstance(i1, list):
            i1, i2 = [i1], [i2]

        args = self.args
        device = args.device
        use_cuda = device >= 0

        # Build vocab
        tok_vocab, char_vocab = build_vocab(
            self.train_path, args.tok_file, args.chr_file,
            args.tok_count_min, args.chr_count_min,
            args.tok_vocab_size, args.chr_vocab_size
        )
        args.chr_vocab_size = len(char_vocab)
        args.tok_vocab_size = len(tok_vocab)

        # Dataset
        input_list = [(1, a, b) for a, b in zip(i1, i2)]
        test_dataset = AVDataset(
            input_list=input_list,
            char_vocab=char_vocab,
            tok_vocab=tok_vocab,
            max_chars_per_word=args.max_chars_per_word,
            max_words_per_sentence=args.max_words_per_sentence,
            max_sentences_per_doc=args.max_sentences_per_doc,
            dont_use_fasttext=args.dont_use_fasttext
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_dataloader_workers
        )

        # Load model
        model = AdHominem(self.args)
        model = load_model_from_disk(self.model_path, model)
        if use_cuda:
            model.to(device=device)

        model.eval()
        all_probas = []
        with torch.no_grad():
            for data in test_loader:
                labels, tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0, sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1 = data_to_device(
                    data, use_cuda=True, device=0)

                embedding0, embedding1 = model(tokens0, tokens1, chars0, chars1, ws_do_mask0, ws_do_mask1, sd_do_mask0,
                                               sd_do_mask1, ws_lens0, ws_lens1, sd_lens0, sd_lens1)
                sims = kernel_fn(euclidean_dist(embedding0, embedding1))
                all_probas.extend(sims.tolist())


        # Convert probs to binary preds
        binary_preds = [1 if p >= threshold else 0 for p in all_probas]
        return all_probas, binary_preds

