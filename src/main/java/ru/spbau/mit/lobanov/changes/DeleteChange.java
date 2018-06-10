package ru.spbau.mit.lobanov.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.tree.ITree;

import java.sql.SQLException;

public class DeleteChange extends AtomicChange {

    private DeleteChange(int nodeType, int parentType, int parentOfParentType, String label) {
        super(nodeType, parentType, parentOfParentType, label);
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.DELETE;
    }

    static DeleteChange fromAction(Delete action) {
        final ITree node = action.getNode();
        return new DeleteChange(
                ChangeUtils.getNodeType(node),
                ChangeUtils.getNodeType(node, 1),
                ChangeUtils.getNodeType(node, 2),
                ChangeUtils.normalize(node.getLabel())
        );
    }

    static DeleteChange fromData(Object[] src, int offset) throws SQLException {
        if ((Integer) src[offset + CHANGE_TYPE_OFFSET] != ChangeType.DELETE.ordinal()) {
            throw new RuntimeException("//todo");
        }
        return new DeleteChange(
                (Integer) src[offset + NODE_TYPE_OFFSET],
                (Integer) src[offset + PARENT_TYPE_OFFSET],
                (Integer) src[offset + PARENT_OF_PARENT_TYPE_OFFSET],
                (String) src[offset + LABEL_OFFSET]
        );
    }
}
