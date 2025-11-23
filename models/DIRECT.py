import collections
import random
import tqdm
import torch as tc
import numpy as np

# MODIFICA: La classe PLM non è più necessaria qui
from .modules import (SoftSelfAttention, GatedFusionExperts,
                      BiasRecommender, SimpleMLP, average)


class DIRECT(tc.nn.Module):
    # MODIFICA: Aggiorniamo il nome della versione per tracciare i modelli salvati
    version = "DIRECT_precomputed"

    def __init__(self, user_num, item_num, aspc_num, aspc_dim,
                 dropout, device, **args):
        super(DIRECT, self).__init__()
        self._cache = {}

        # --- INIZIO MODIFICA SOSTANZIALE ---
        # RIMUOVIAMO COMPLETAMENTE L'ENCODER DAL MODELLO.
        # La variabile 'plm' non viene più passata.
        # self.bert = PLM(plm, dropout)

        # La dimensione dell'embedding ora è un parametro fisso,
        word_dim = 2304
        # --- FINE MODIFICA SOSTANZIALE ---

        # Il resto dell'architettura rimane identico, perché dipende solo da 'word_dim'.
        self.sentiment_tagging = SimpleMLP(word_dim, word_dim // 4, 1, min(dropout * 2, 0.5))

        self.item_embs = tc.nn.Parameter(tc.zeros((item_num + 1, word_dim)))
        tc.nn.init.xavier_uniform_(self.item_embs)
        self.review_agg = SoftSelfAttention(word_dim, "first", dropout=0.1)
        self.history_agg = SoftSelfAttention(word_dim, "mean", dropout=0.1)
        self.interest_proj = GatedFusionExperts([word_dim, word_dim], aspc_dim,
                                                reduction=8, exploration=4,
                                                dropout=dropout)
        self.interest_norm = tc.nn.BatchNorm1d(aspc_num + 1)

        self.mention_proj = tc.nn.Linear(word_dim, aspc_dim)

        # La logica degli aspetti rimane, ma le loss custom che la usano sono disattivate
        self.asp_free = tc.tensor([0.] + [1.] * aspc_num).reshape(1, 1, -1).to(device)
        self.asp_eye = tc.eye(aspc_num + 1).to(device)
        self.asp_embs = tc.nn.Parameter(tc.ones((aspc_dim, aspc_num + 1)))
        tc.nn.init.xavier_uniform_(self.asp_embs)

        self.bias = BiasRecommender(user_num, item_num)
        self.lossfn = tc.nn.MSELoss()
        self.device = device
        self.to(device)

    # --- MODIFICA: La funzione 'prepare_item_embedding' non serve più ---
    def prepare_item_embedding(self, *args, **kwargs):
        # La lasciamo vuota per non rompere la chiamata nello script di training
        print("INFO: 'prepare_item_embedding' non è più necessaria con gli embedding pre-calcolati. Saltata.")
        pass

    def forward(self, uid, iid, user_doc_embedding, user_doc_mask,
                item_doc_embedding, item_doc_mask, user_hist_ids, **args):
        """
        La firma è cambiata: ora riceve direttamente i tensori degli embedding.
        """
        self._cache.clear()
        uid, iid = uid.to(self.device), iid.to(self.device)
        user_hist = user_hist_ids.to(self.device)

        # --- INIZIO MODIFICA: Usiamo gli embedding pre-calcolati ---
        # Spostiamo gli embedding e le maschere (che sono già tensori) sulla GPU
        user_emb = user_doc_embedding.to(self.device)
        user_mask = user_doc_mask.to(self.device)
        item_emb = item_doc_embedding.to(self.device)
        item_mask = item_doc_mask.to(self.device)
        # --- FINE MODIFICA ---

        sentiment = self._sentiment_analysis(item_emb)
        gate = (self._mention(item_emb) * self._interest(user_emb, user_mask, user_hist)).sum(axis=-1) * item_mask
        major_score = average(sentiment * gate, gate).squeeze()
        bias_score = self.bias(uid, iid)

        # La logica delle loss custom (attualmente disattivata) va adattata se la si riattiva.
        return bias_score + major_score

        # --- MODIFICA: Il metodo _encode non esiste più ---

    def _sentiment_analysis(self, embs):
        return tc.tanh(self.sentiment_tagging(embs)).squeeze()

    def _interest(self, doc_emb, doc_mask, hist):
        # NOTA: Questa parte usa self.item_embs, che sono embedding addestrabili,
        # non quelli pre-calcolati. Questo è corretto per la logica del modello.
        domains = [self.review_agg(doc_emb, doc_mask),
                   self.history_agg(self.item_embs[hist.long()], tc.where(hist >= 1, 1.0, 0.0))]
        user_emb = self.interest_proj(*domains)
        interest = self.interest_norm(tc.mm(user_emb, self.asp_embs)).unsqueeze(1)
        return tc.sigmoid(interest) * self.asp_free

    def _mention(self, iemb):
        bs, ts, dim = iemb.shape
        mention = self.mention_proj(iemb.reshape(-1, dim))
        mention = tc.nn.functional.normalize(mention, dim=-1)
        aspects = tc.nn.functional.normalize(self.asp_embs, dim=0)
        mention = tc.mm(mention, aspects).reshape(bs, ts, -1)
        return tc.softmax(mention, -1)

    def compute_loss(self, ytrue, ypred):
        ypred = ypred.to(self.device).float()
        ytrue = ytrue.to(self.device).float()
        loss = self.lossfn(ypred, ytrue)
        # Ricorda che le loss custom sono disattivate e richiederebbero un lavoro
        # di adattamento per funzionare con gli embedding pre-calcolati.
        # if self.training:
        #     loss = loss + ...
        return loss
