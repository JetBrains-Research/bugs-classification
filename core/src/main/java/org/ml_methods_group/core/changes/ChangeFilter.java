package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.actions.model.Update;

public interface ChangeFilter {
    Resolution accept(Update update);
    Resolution accept(Insert insert);
    Resolution accept(Move move);
    Resolution accept(Delete delete);

    enum Resolution {
        ACCEPT, ACCEPT_WITHOUT_LABEL, REJECT
    }
}
