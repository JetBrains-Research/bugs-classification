package org.ml_methods_group.core.changes;

public class UpdateChange implements AtomicChange {

    private final NodeType nodeType;
    private final NodeType parentType;
    private final NodeType parentOfParentType;
    private final String label;
    private final String oldLabel;

    public UpdateChange(NodeType nodeType, NodeType parentType, NodeType parentOfParentType,
                        String label, String oldLabel) {
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.label = label;
        this.oldLabel = oldLabel;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.UPDATE;
    }

    @Override
    public NodeType getNodeType() {
        return nodeType;
    }

    @Override
    public String getLabel() {
        return label;
    }

    @Override
    public NodeType getParentType() {
        return parentType;
    }

    @Override
    public NodeType getParentOfParentType() {
        return parentOfParentType;
    }

    @Override
    public String getOldLabel() {
        return oldLabel;
    }
}
