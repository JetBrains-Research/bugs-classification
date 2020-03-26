import torch
from utils import fuzzy_pr_auc_score


class AttentionBiGRUClassifier(torch.nn.Module):

    def __init__(self, embedding_dim, tokens_per_change, hidden_dim, atomic_dim, vocab_size, labels_set_size,
                 padding_idx):
        super(AttentionBiGRUClassifier, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.fc1 = torch.nn.Linear(embedding_dim * tokens_per_change, atomic_dim)
        self.gru = torch.nn.GRU(atomic_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = torch.nn.Linear(hidden_dim * 2, 1)
        self.fc2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_dim, labels_set_size)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.embedding_dim = embedding_dim
        self.tokens_per_change = tokens_per_change

    def get_embeddings(self, tokens, real_lengths):
        token_embeds = self.dropout(self.embeddings(tokens))
        token_groups = token_embeds.view(token_embeds.shape[0],
                                         token_embeds.shape[1] // self.tokens_per_change,
                                         self.tokens_per_change * self.embedding_dim)
        changes_embeds = self.fc1(token_groups)
        real_lengths //= self.tokens_per_change
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(changes_embeds,
                                                                real_lengths,
                                                                batch_first=True,
                                                                enforce_sorted=False)
        gru_outs, _ = self.gru(packed_embeds)
        gru_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_outs, batch_first=True)
        weights = torch.functional.F.softmax(self.attention(gru_outs).squeeze(-1), dim=1)
        hidden = torch.sum(weights.unsqueeze(-1) * gru_outs, dim=1)
        edit_script_embeddings = self.act(self.fc2(hidden))
        return edit_script_embeddings

    def forward(self, tokens, real_lengths):
        edit_script_embeddings = self.get_embeddings(tokens, real_lengths)
        label_preds = self.out(self.dropout(edit_script_embeddings))
        return label_preds.squeeze(0)

    def test(self, X, X_real_lengths, y_true, labels):
        preds = self.forward(X, X_real_lengths)
        preds = torch.functional.F.softmax(preds, dim=1)
        return fuzzy_pr_auc_score(y=y_true,
                                  y_pred=preds.cpu().detach().numpy(),
                                  labels=labels)
