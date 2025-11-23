import json
import os
import re
import time
import pickle
import collections
import random
import tqdm
import torch as tc
import numpy as np

from .tokenizer import Tokenizer
from .lookup import LookupTable


class CorpusSearchIndex:
    def __init__(self, file_path, cache_freq=5, sampling=None):
        assert os.path.exists(file_path)
        self.datafile = file_path
        self.cache_freq = cache_freq
        self.lookup, self._numrow = [0], 0
        with open(file_path, encoding="utf8") as f:
            while self._numrow != sampling:
                row = f.readline()
                if len(row) == 0:
                    break
                self._numrow += 1
                if self._numrow % cache_freq == 0:
                    self.lookup.append(f.tell())

    def __iter__(self):
        with open(self.datafile, encoding="utf8") as f:
            for row in f:
                yield row.strip()

    def __len__(self):
        return self._numrow

    def __getitem__(self, index):
        cacheid = index // self.cache_freq
        with open(self.datafile, encoding="utf8") as f:
            f.seek(self.lookup[cacheid])
            for idx, row in enumerate(f, cacheid * self.cache_freq):
                if idx == index:
                    return row.strip()
        raise IndexError("Index %d is out of boundary" % index)


class DocumentDataset(tc.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_len, keep_head, cache_freq):
        super(tc.utils.data.Dataset).__init__()
        self.tokenizer = Tokenizer(tokenizer, max_len, keep_head)
        self.documents = CorpusSearchIndex(file_path, cache_freq)

    def __getitem__(self, idx):
        doc = self.documents[idx].replace("\t", " ").strip()
        doc_id, doc_mask = self.tokenizer.transform(doc.split())
        return np.array(doc_id), np.array(doc_mask)

    def __len__(self):
        return len(self.documents)


class MetaIndex:
    def __init__(self, root, tokenizer, num_sent, len_sent,
                 num_hist, len_doc, users=None, items=None, drop_stopword=False, keep_head=0.7, cache_freq=5):

        self.num_hist = num_hist
        self.root = root
        self.users = users if users else LookupTable.from_txt(os.path.join(root, "user_idx.txt"))
        self.items = items if items else LookupTable.from_txt(os.path.join(root, "item_idx.txt"))

        self.user_stars = CorpusSearchIndex(os.path.join(root, "user_score.txt"), cache_freq)
        self.user_hist = CorpusSearchIndex(os.path.join(root, "user_history.txt"), cache_freq)
        self.item_hist = CorpusSearchIndex(os.path.join(root, "item_history.txt"), cache_freq)

        # --- MODIFICA: Ripristiniamo il Memory Mapping ---

        print("INFO: Inizio caricamento degli embedding consolidati in RAM...")

        user_embeddings_file = os.path.join(root, "all_user_embeddings.npy")
        user_masks_file = os.path.join(root, "all_user_masks.npy")
        item_embeddings_file = os.path.join(root, "all_item_embeddings.npy")
        item_masks_file = os.path.join(root, "all_item_masks.npy")

        # Rimuoviamo 'mmap_mode' per caricare l'intero array in memoria.
        self.user_embeddings = np.load(user_embeddings_file)
        self.user_masks = np.load(user_masks_file)
        self.item_embeddings = np.load(item_embeddings_file)
        self.item_masks = np.load(item_masks_file)

        print("INFO: Embedding caricati in RAM con successo.")

        print("INFO: Embedding pronti per l'accesso tramite memory map.")
        # --- FINE MODIFICA ---

    def get_feed_dict(self, uid, iid, current_review=""):
        # Questa funzione rimane identica alla versione precedente.
        # L'accesso a self.user_embeddings[uid] sarà ora un accesso alla RAM,
        # ancora più veloce del memory mapping.

        if isinstance(uid, str):
            uid = self.users[uid]
        if isinstance(iid, str):
            iid = self.items[iid]

        rating_hist = self._get_history(self.user_stars[uid], float, False)
        user_hist = self._get_history(self.user_hist[uid], self.items.__getitem__, True)
        item_hist = self._get_history(self.item_hist[iid], self.users.__getitem__, True)

        user_doc_embedding = self.user_embeddings[uid].copy()
        user_doc_mask = self.user_masks[uid].copy()

        item_doc_embedding = self.item_embeddings[iid].copy()
        item_doc_mask = self.item_masks[iid].copy()

        return {"uid": uid,
                "iid": iid,
                "user_hist_ids": user_hist,
                "item_hist_ids": item_hist,
                "user_hist_rate": rating_hist,

                "user_doc_embedding": user_doc_embedding,
                "user_doc_mask": user_doc_mask,
                "item_doc_embedding": item_doc_embedding,
                "item_doc_mask": item_doc_mask,

                "current_ids": -1, "current_mask": -1, "user_doc_ids": -1,
                "user_sent_ids": -1, "user_sent_mask": -1, "item_doc_ids": -1,
                "item_sent_ids": -1, "item_sent_mask": -1,
                }

        # --------------------------------------------------------------------

    # Le funzioni _get_history, _get_sent_reviews, _get_doc_reviews rimangono,
    # anche se le ultime due non verranno più chiamate.

    def _get_history(self, data, func, add_bias=True, padding=0):
        hist = [func(_) for _ in data.split("\t")[:self.num_hist] if len(_) > 0]
        if add_bias:
            hist = [_ + 1 for _ in hist]
        return np.array(hist + [padding] * (self.num_hist - len(hist)))


    def _get_sent_reviews(self, reviews):
        # QUESTA FUNZIONE È OBSOLETA E NON DOVREBBE ESSERE CHIAMATA.
        # Se viene chiamata, significa che c'è un errore nella logica del programma.
        raise NotImplementedError(
            "La funzione '_get_sent_reviews' è stata deprecata con l'introduzione degli embedding pre-calcolati. "
            "Non dovrebbe essere mai chiamata."
        )

    def _get_doc_reviews(self, reviews):
        # QUESTA FUNZIONE È OBSOLETA E NON DOVREBBE ESSERE CHIAMATA.
        raise NotImplementedError(
            "La funzione '_get_doc_reviews' è stata deprecata con l'introduzione degli embedding pre-calcolati. "
            "Non dovrebbe essere mai chiamata."
        )


class Dataset(tc.utils.data.Dataset):
    def __init__(self, root, subset, format_, metaset, cache_freq, sampling,
                 paths=None, split="train", setup="DEFAULT"):

        super(tc.utils.data.Dataset).__init__()
        self.keys, self.meta = format_, metaset
        self.users, self.items = self.meta.users, self.meta.items
        self.data = CorpusSearchIndex(root + r"/%s.json" % subset, cache_freq, sampling)

        self.split = split
        self.setup = setup

        if setup == "BPR":

            self.positive = {}
            self.interacted_train = {}
            self.interacted_val = {}

            # Set di item positivi per gli utenti del set corrente
            self.positive = self.interactions(root + "/train.json")

            # Item con cui l'utente ha interagito in validation
            if split == "test":
                self.interacted_val = self.interactions(paths['valid'] + "/train.json")

            # Item con cui l'utente ha interagito in train
            if split == "test" or split == "valid":
                self.interacted_train = self.interactions(paths['train'] + "/train.json")

    @staticmethod
    def interactions(filepath):
        with open(filepath, mode="r", newline="") as f:
            positive = {}
            for line in f:
                row = json.loads(line)

                user = row['reviewerID']
                item = row['asin']

                if user not in positive:
                    positive[user] = set()
                positive[user].add(item)
        return positive

    def get_negative(self, user):
        allItems = self.items.id2str
        negatives = [x for x in allItems if x not in self.positive[user]]
        return random.choice(negatives)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for row in self.data:
            user, item, review, score = self.get_record(row)

            # Se in setup BPR, si comporta diversamente
            if self.setup == "BPR" and self.split == "train":
                neg_item = self.get_negative(user)
                yield (self.users[user], item, review, 1), (self.items[item], neg_item, "", 0)

            recommend = 1. if score >= 4.0 else 0.
            yield self.users[user], self.items[item], review, recommend

    def __getitem__(self, idx):
        user, item, review, score = self.get_record(self.data[idx])

        # Se in setup BPR, si comporta diversamente
        if self.setup == "BPR" and self.split == "train":
            positive = self.meta.get_feed_dict(user, item, review)
            positive.update({"score": score, "recommend": 1})

            neg_item = self.get_negative(user)
            negative = self.meta.get_feed_dict(user, neg_item, "")
            negative.update({"score": 1, "recommend": 0})
            return positive, negative

        recommend = 1. if score >= 4.0 else 0.
        feed = self.meta.get_feed_dict(user, item, review)
        feed.update({"score": score, "recommend": recommend})
        return feed

    @property
    def num_user(self):
        return len(self.users)

    @property
    def num_item(self):
        return len(self.items)

    def get_record(self, row):
        row = json.loads(row)
        return (row[_] for _ in self.keys)

