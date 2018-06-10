package ru.spbau.mit.lobanov.changes;

import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.tree.ITree;

import java.sql.SQLException;

public class InsertChange extends AtomicChange {

    private InsertChange(int nodeType, int parentType, int parentOfParentType, String label) {
        super(nodeType, parentType, parentOfParentType, label);
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.INSERT;
    }

    static InsertChange fromAction(Insert action) {
        final ITree node = action.getNode();
        final ITree parent = node.getParent();
        return new InsertChange(
                ChangeUtils.getNodeType(node),
                ChangeUtils.getNodeType(parent),
                ChangeUtils.getNodeType(parent, 1),
                ChangeUtils.normalize(node.getLabel())
        );
    }

    static InsertChange fromData(Object[] src, int offset) throws SQLException {
        if ((Integer) src[offset + CHANGE_TYPE_OFFSET] != ChangeType.INSERT.ordinal()) {
            throw new RuntimeException("//todo");
        }
        return new InsertChange(
                (Integer) src[offset + NODE_TYPE_OFFSET],
                (Integer) src[offset + PARENT_TYPE_OFFSET],
                (Integer) src[offset + PARENT_OF_PARENT_TYPE_OFFSET],
                (String) src[offset + LABEL_OFFSET]
        );
    }
}
