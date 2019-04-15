# -*- coding: utf-8 -*-

PAD_TOKEN = '<PAD>'

class SequenceAlignment(object):
    def __init__(self):
        self.REPLACE_TOKEN = 0
        self.ADD_TOKEN = 1
        self.REMOVE_TOKEN = 2
        self.EQUALS = 3 
        
        self.REPLACE_VEC = [1, 0, 0, 0]
        self.ADD_VEC = [0, 1, 0, 0]
        self.REMOVE_VEC = [0, 0, 1, 0]
        self.EQUAL_VEC = [0, 0, 0, 1]

    def set_cell_cost(self, i, j, cost_matrix, path_matrix, are_tokens_equal):
        take_both = cost_matrix[i - 1][j - 1] + 1 * (1 if are_tokens_equal else -1)
        take_prev = cost_matrix[i][j - 1] - 2
        take_upd = cost_matrix[i - 1][j] - 2
        cost_matrix[i][j] = max(take_both, take_prev, take_upd)
        if take_prev == cost_matrix[i][j]:
            path_matrix[i][j] = (i, j - 1)
        elif take_upd == cost_matrix[i][j]:
            path_matrix[i][j] = (i - 1, j)
        else:
            path_matrix[i][j] = (i - 1, j - 1)
        
    def get_aligment_cost_matrix(self, prev_seq, upd_seq):
        prev_len = len(prev_seq) + 1
        upd_len = len(upd_seq) + 1
        cost_matrix = []
        path_matrix = []
        for i in range(upd_len):
            cost_matrix.append([])
            path_matrix.append([])
            for j in range(prev_len):
                cost_matrix[i].append([])
                path_matrix[i].append([])
      
        for i in range(upd_len):
            cost_matrix[i][0] = -2 * i
        for j in range(prev_len):
            cost_matrix[0][j] = -2 * j
        
        diag_len = min(prev_len, upd_len)
        for diag in range(1, diag_len):
            self.set_cell_cost(diag, diag, cost_matrix, path_matrix, upd_seq[diag - 1] == prev_seq[diag - 1])
            for i in range(diag + 1, upd_len):
                self.set_cell_cost(i, diag, cost_matrix, path_matrix, upd_seq[i - 1] == prev_seq[diag - 1])
            for j in range(diag + 1, prev_len):
                self.set_cell_cost(diag, j, cost_matrix, path_matrix, upd_seq[diag - 1] == prev_seq[j - 1])
        return cost_matrix, path_matrix

    def get_alignment(self, prev_seq, upd_seq):
        cost_matrix, path_matrix = self.get_aligment_cost_matrix(prev_seq, upd_seq)
        alignment = []
        i = len(upd_seq)
        j = len(prev_seq)
        while i > 0 and j > 0:
            prev_i, prev_j = path_matrix[i][j]
            if prev_i == i:
                alignment.insert(0, self.REMOVE_TOKEN)
            elif prev_j == j:
                alignment.insert(0, self.ADD_TOKEN)
            elif upd_seq[i - 1] == prev_seq[j - 1]:
                alignment.insert(0, self.EQUALS)
            else:
                alignment.insert(0, self.REPLACE_TOKEN)
            i = prev_i
            j = prev_j
        return alignment

    def prepare_row(self, alignment, prev_tokens, upd_tokens):
        edit_vector = []
        prev_padded = []
        upd_padded = []
        prev_token_id = 0
        upd_token_id = 0
        for symbol in alignment:
            if symbol == self.REPLACE_TOKEN:
                edit_vector.append(self.REPLACE_VEC)
                prev_padded.append(prev_tokens[prev_token_id])
                prev_token_id += 1
                upd_padded.append(upd_tokens[upd_token_id])
                upd_token_id += 1
            elif symbol == self.ADD_TOKEN:
                edit_vector.append(self.ADD_VEC)
                prev_padded.append(PAD_TOKEN)
                upd_padded.append(upd_tokens[upd_token_id])
                upd_token_id += 1
            elif symbol == self.REMOVE_TOKEN:
                edit_vector.append(self.REMOVE_VEC)
                prev_padded.append(prev_tokens[prev_token_id])
                prev_token_id += 1
                upd_padded.append(PAD_TOKEN)
            else:
                edit_vector.append(self.EQUAL_VEC)
                prev_padded.append(prev_tokens[prev_token_id])
                prev_token_id += 1
                upd_padded.append(upd_tokens[upd_token_id])
                upd_token_id += 1
        assert len(edit_vector) == len(prev_padded)
        assert len(edit_vector) == len(upd_padded)
        return edit_vector, prev_padded, upd_padded
    
    def prepare_for_nn_input(self, prev_seqs, upd_seqs):
        assert len(prev_seqs) == len(upd_seqs)
        n_samples = len(prev_seqs)
        edit_vecs = []
        prev_pad_vecs = []
        upd_pad_vecs = []
        for sample_id in range(n_samples):
            alignment = self.get_alignment(prev_seqs[sample_id], upd_seqs[sample_id])
            edit, prev, upd = self.prepare_row(alignment, prev_seqs[sample_id], upd_seqs[sample_id]) 
            edit_vecs.append(edit)
            prev_pad_vecs.append(prev)
            upd_pad_vecs.append(upd)
        return edit_vecs, prev_pad_vecs, upd_pad_vecs