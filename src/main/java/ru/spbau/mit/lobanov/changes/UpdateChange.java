package ru.spbau.mit.lobanov.changes;

import com.github.gumtreediff.actions.model.Update;
import com.github.gumtreediff.tree.ITree;
import ru.spbau.mit.lobanov.database.Table;

import java.sql.SQLException;

public class UpdateChange extends AtomicChange {
    private final String oldLabel;

    private UpdateChange(int nodeType, int parentType, int parentOfParentType, String label, String oldLabel) {
        super(nodeType, parentType, parentOfParentType, label);
        this.oldLabel = ChangeUtils.normalize(oldLabel);
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.UPDATE;
    }

    @Override
    public void storeData(Object[] dst, int offset) {
        super.storeData(dst, offset);
        dst[offset + OLD_LABEL_OFFSET] = oldLabel;
    }

    @Override
    public String getOldLabel() {
        return oldLabel;
    }

    static UpdateChange fromData(Object[] src, int offset) throws SQLException {
        if ((Integer) src[offset + CHANGE_TYPE_OFFSET] != ChangeType.UPDATE.ordinal()) {
            throw new RuntimeException("//todo");
        }
        return new UpdateChange(
                (Integer) src[offset + NODE_TYPE_OFFSET],
                (Integer) src[offset + PARENT_TYPE_OFFSET],
                (Integer) src[offset + PARENT_OF_PARENT_TYPE_OFFSET],
                (String) src[offset + LABEL_OFFSET],
                (String) src[offset + OLD_LABEL_OFFSET]
        );
    }

    static UpdateChange fromAction(Update action) {
        final ITree node = action.getNode();
        return new UpdateChange(
                ChangeUtils.getNodeType(node),
                ChangeUtils.getNodeType(node, 1),
                ChangeUtils.getNodeType(node, 2),
                ChangeUtils.normalize(action.getValue()),
                ChangeUtils.normalize(node.getLabel())
        );
    }
}
