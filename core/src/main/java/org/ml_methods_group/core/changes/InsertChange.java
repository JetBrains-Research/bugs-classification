package org.ml_methods_group.core.changes;

public class InsertChange implements AtomicChange {

    private final NodeType nodeType;
    private final NodeType parentType;
    private final NodeType parentOfParentType;
    private final String label;

    public InsertChange(NodeType nodeType, NodeType parentType, NodeType parentOfParentType, String label) {
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.label = label;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.INSERT;
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
}
