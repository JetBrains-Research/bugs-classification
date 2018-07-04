package org.ml_methods_group.core.changes;

public interface AtomicChange {

    ChangeType getChangeType();

    default NodeType getNodeType() {
        return NodeType.NONE;
    }

    default NodeType getParentType() {
        return NodeType.NONE;
    }

    default NodeType getParentOfParentType() {
        return NodeType.NONE;
    }

    default String getLabel() {
        return "";
    }

    default NodeType getOldParentType() {
        return NodeType.NONE;
    }

    default NodeType getOldParentOfParentType() {
        return NodeType.NONE;
    }

    default String getOldLabel() {
        return "";
    }
}
