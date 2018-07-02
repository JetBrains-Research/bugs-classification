package org.ml_methods_group.core.changes;

public class DeleteChange implements AtomicChange {

    private final int nodeType;
    private final String label;
    private final int oldParentType;
    private final int oldParentOfParentType;

    public DeleteChange(int nodeType, String label, int oldParentType, int oldParentOfParentType) {
        this.nodeType = nodeType;
        this.label = label;
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.DELETE;
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
    public int getOldParentType() {
        return oldParentType;
    }

    @Override
    public int getOldParentOfParentType() {
        return oldParentOfParentType;
    }
}
