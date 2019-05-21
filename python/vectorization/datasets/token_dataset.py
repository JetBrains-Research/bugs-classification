# -*- coding: utf-8 -*-

import json
import pickle
import codecs

from alignment import SequenceAlignment

class TokenDataset(object):
    def __init__(self, data_path, edits_path, prevs_path, upds_path, mark_out_path = None, mark_in_path = None):
        self.data_path = data_path
        self.alignment = SequenceAlignment()
        self.edits_path = edits_path
        self.prevs_path = prevs_path
        self.upds_path = upds_path
        self.mark_out_path = mark_out_path
        self.mark_in_path = mark_in_path
    
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
        marks = []
        for json_row in json_lines:
            prev_data_tokens.append(json_row['PrevCodeChunkTokens'])
            upd_data_tokens.append(json_row['UpdatedCodeChunkTokens'])
            marks.append(json_row['Id'])
        return prev_data_tokens, upd_data_tokens, marks
    
    def prepare_token_dataset(self):
        json_lines = self.read_jsonl()
        prev_tokens, upd_tokens, marks = self.get_tokens_from_jsonl(json_lines)
        edit_vecs, prev_pad_tokens, upd_pad_tokens = self.alignment.prepare_for_nn_input(prev_tokens, upd_tokens)
        self.save(self.edits_path, edit_vecs)
        self.save(self.prevs_path, prev_pad_tokens)
        self.save(self.upds_path, upd_pad_tokens)
        if self.mark_in_path is not None:
            mark2id, _ = self.create_marks_dict(self.mark_in_path)
            mark_ids = []
            for mark in marks:
                if mark in mark2id:
                    mark_ids.append(mark2id[mark])
                else:
                    mark_ids.append(0)
            self.save(self.mark_out_path, mark_ids)

    def save(self, save_path, object_to_save):
        with open(save_path, 'wb') as f:
            pickle.dump(object_to_save, f)
            
    def create_marks_dict(self, mark_path): 
        mark2id = {} 
        id2mark = {} 
        with open(mark_path, 'r') as mark_file: 
            for line_id, line in enumerate(mark_file): 
                line = line.replace('\n', '') 
                mark2id[line] = line_id + 1
                id2mark[line_id + 1] = line 
        return mark2id, id2mark           