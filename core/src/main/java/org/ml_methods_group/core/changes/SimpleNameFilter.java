package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.actions.model.Update;

import static org.ml_methods_group.core.entities.NodeType.SIMPLE_NAME;

public class SimpleNameFilter implements ChangeFilter {
    @Override
    public boolean accept(Update update) {
        return update.getNode().getType() != SIMPLE_NAME.ordinal() || Utils.isMethodName(update.getNode());
    }

    @Override
    public boolean accept(Insert insert) {
        return true;
    }

    @Override
    public boolean accept(Move move) {
        return true;
    }

    @Override
    public boolean accept(Delete delete) {
        return true;
    }
}
