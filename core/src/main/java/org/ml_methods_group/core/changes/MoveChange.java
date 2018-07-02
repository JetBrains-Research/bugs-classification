package org.ml_methods_group.core.changes;

public class MoveChange implements AtomicChange {

    private final int nodeType;
    private final String label;
    private final int parentType;
    private final int parentOfParentType;
    private final int oldParentType;
    private final int oldParentOfParentType;

    public MoveChange(int nodeType, String label, int parentType, int parentOfParentType,
                      int oldParentType, int oldParentOfParentType) {
        this.nodeType = nodeType;
        this.label = label;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.MOVE;
    }

    @Override
    public int getNodeType() {
        return nodeType;
    }

    @Override
    public String getLabel() {
        return label;
    }

    @Override
    public int getParentType() {
        return parentType;
    }

    @Override
    public int getParentOfParentType() {
        return parentOfParentType;
    }

    @Override
    public int getOldParentType() {
        return oldParentType;
    }

    @Override
    public int getOldParentOfParentType() {
        return oldParentOfParentType;
    }
}
