package org.ml_methods_group.core.changes;

public class MoveChange implements AtomicChange {

    private final NodeType nodeType;
    private final NodeType parentType;
    private final NodeType parentOfParentType;
    private final NodeType oldParentType;
    private final NodeType oldParentOfParentType;
    private final String label;

    public MoveChange(NodeType nodeType, NodeType parentType, NodeType parentOfParentType,
                      NodeType oldParentType, NodeType oldParentOfParentType, String label) {
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
        this.label = label;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.MOVE;
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
    public NodeType getOldParentType() {
        return oldParentType;
    }

    @Override
    public NodeType getOldParentOfParentType() {
        return oldParentOfParentType;
    }
}
