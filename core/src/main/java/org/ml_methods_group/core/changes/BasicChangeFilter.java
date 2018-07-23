package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.actions.model.Update;
import com.github.gumtreediff.tree.ITree;

import static org.ml_methods_group.core.entities.NodeType.*;

public class BasicChangeFilter implements ChangeFilter {
    @Override
    public boolean accept(Update update) {
        return !isComment(update.getNode()) &&
                (update.getNode().getType() != SIMPLE_NAME.ordinal() || Utils.isMethodName(update.getNode()));
    }

    @Override
    public boolean accept(Insert insert) {
        return !isComment(insert.getNode());
    }

    @Override
    public boolean accept(Move move) {
        return !isComment(move.getNode());
    }

    @Override
    public boolean accept(Delete delete) {
        return !isComment(delete.getNode());
    }

    private static boolean isComment(ITree node) {
        final int type = node.getType();
        return type == JAVADOC.ordinal() || type == LINE_COMMENT.ordinal() || type == BLOCK_COMMENT.ordinal();
    }
}
