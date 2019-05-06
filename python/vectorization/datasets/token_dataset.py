# -*- coding: utf-8 -*-

import json
import pickle
import codecs

from alignment import SequenceAlignment

class TokenDataset(object):
    def __init__(self, data_path, edits_path, prevs_path, upds_path):
        self.data_path = data_path
        self.alignment = SequenceAlignment()
        self.edits_path = edits_path
        self.prevs_path = prevs_path
        self.upds_path = upds_path
    
    def read_jsonl(self):
        with codecs.open(self.data_path, mode = 'r', encoding = "utf-8-sig") as data_file:
            data_rows = data_file.read().split('\n')
            json_lines = []
            for row_id, row in enumerate(data_rows):
                try:
                    json_row = json.loads(row)
                    json_lines.append(json_row)
                except json.JSONDecodeError:
                    pass
        return json_lines

    def get_tokens_from_jsonl(self, json_lines):
        prev_data_tokens = []
        upd_data_tokens = []
        for json_row in json_lines:
            prev_data_tokens.append(json_row['PrevCodeChunkTokens'])
            upd_data_tokens.append(json_row['UpdatedCodeChunkTokens'])
        return prev_data_tokens, upd_data_tokens
    
    def prepare_token_dataset(self):
        json_lines = self.read_jsonl()
        prev_tokens, upd_tokens = self.get_tokens_from_jsonl(json_lines)
        edit_vecs, prev_pad_tokens, upd_pad_tokens = self.alignment.prepare_for_nn_input(prev_tokens, upd_tokens)
        self.save(self.edits_path, edit_vecs)
        self.save(self.prevs_path, prev_pad_tokens)
        self.save(self.upds_path, upd_pad_tokens)

    def save(self, save_path, object_to_save):
        with open(save_path, 'wb') as f:
            pickle.dump(object_to_save, f)