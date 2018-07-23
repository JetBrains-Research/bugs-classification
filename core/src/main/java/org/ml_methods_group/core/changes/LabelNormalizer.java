package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.actions.model.Update;

public interface LabelNormalizer {
    String normalize(String label, Insert insert);
    String normalize(String label, Move move);
    String normalize(String label, Update update);
    String normalize(String label, Delete delete);
}
