package ru.spbau.mit.lobanov.changes;

import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.tree.ITree;
import ru.spbau.mit.lobanov.database.Table;

import java.sql.SQLException;

public class MoveChange extends AtomicChange {
    private final int oldParentType;
    private final int oldParentOfParentType;

    private MoveChange(int nodeType, int parentType, int parentOfParentType, String label,
                       int oldParentType, int oldParentOfParentType) {
        super(nodeType, parentType, parentOfParentType, label);
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
    }

    @Override
    public void storeData(Object[] dst, int offset) {
        super.storeData(dst, offset);
        dst[offset + OLD_PARENT_TYPE_OFFSET] = oldParentType;
        dst[offset + OLD_PARENT_OF_PARENT_TYPE_OFFSET] = oldParentOfParentType;
    }

    @Override
    public int getOldParentType() {
        return oldParentType;
    }

    @Override
    public int getOldParentOfParentType() {
        return oldParentOfParentType;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.MOVE;
    }

    static MoveChange fromAction(Move action) {
        final ITree node = action.getNode();
        final ITree parent = action.getParent();
        return new MoveChange(
                ChangeUtils.getNodeType(node),
                ChangeUtils.getNodeType(parent),
                ChangeUtils.getNodeType(parent, 1),
                ChangeUtils.normalize(node.getLabel()),
                ChangeUtils.getNodeType(node, 1),
                ChangeUtils.getNodeType(node, 2)
        );
    }

    static MoveChange fromData(Object[] src, int offset) throws SQLException {
        if ((Integer) src[offset + CHANGE_TYPE_OFFSET] != ChangeType.MOVE.ordinal()) {
            throw new RuntimeException("//todo");
        }
        return new MoveChange(
                (Integer) src[offset + NODE_TYPE_OFFSET],
                (Integer) src[offset + PARENT_TYPE_OFFSET],
                (Integer) src[offset + PARENT_OF_PARENT_TYPE_OFFSET],
                (String) src[offset + LABEL_OFFSET],
                (Integer) src[offset + OLD_PARENT_TYPE_OFFSET],
                (Integer) src[offset + OLD_PARENT_OF_PARENT_TYPE_OFFSET]
        );
    }
}
