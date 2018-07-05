package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.ChangeType;
import org.ml_methods_group.core.changes.NodeType;

public class ChangeCodeWrapper {
    private final long code;
    private final int encodingType;
    private final ChangeType changeType;
    private final NodeType nodeType;
    private final NodeType parentType;
    private final NodeType parentOfParentType;
    private final NodeType oldParentType;
    private final NodeType oldParentOfParentType;
    private final String label;
    private final String oldLabel;

    public ChangeCodeWrapper(long code, int encodingType, ChangeType changeType,
                             NodeType nodeType, NodeType parentType, NodeType parentOfParentType,
                             NodeType oldParentType, NodeType oldParentOfParentType,
                             String label, String oldLabel) {
        this.code = code;
        this.encodingType = encodingType;
        this.changeType = changeType;
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
        this.label = label;
        this.oldLabel = oldLabel;
    }

    public long getCode() {
        return code;
    }

    public int getEncodingType() {
        return encodingType;
    }

    public NodeType getNodeType() {
        return nodeType;
    }

    public NodeType getParentType() {
        return parentType;
    }

    public NodeType getParentOfParentType() {
        return parentOfParentType;
    }

    public NodeType getOldParentType() {
        return oldParentType;
    }

    public NodeType getOldParentOfParentType() {
        return oldParentOfParentType;
    }

    public String getLabel() {
        return label;
    }

    public String getOldLabel() {
        return oldLabel;
    }

    public ChangeType getChangeType() {
        return changeType;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ChangeCodeWrapper that = (ChangeCodeWrapper) o;
        return code == that.code && encodingType == that.encodingType;
    }

    @Override
    public int hashCode() {
        int result = (int) (code ^ (code >>> 32));
        result = 31 * result + encodingType;
        return result;
    }
}
