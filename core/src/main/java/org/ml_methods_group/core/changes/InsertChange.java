package org.ml_methods_group.core.changes;

public class InsertChange implements AtomicChange {

    private final int nodeType;
    private final String label;
    private final int parentType;
    private final int parentOfParentType;

    public InsertChange(int nodeType, String label, int parentType, int parentOfParentType) {
        this.nodeType = nodeType;
        this.label = label;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.INSERT;
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
}
