package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

import static org.ml_methods_group.core.entities.ChangeType.*;
import static org.ml_methods_group.core.entities.NodeType.NONE;

@DataClass(defaultStorageName = "changes")
public class CodeChange {
    @DataField
    private final ChangeType changeType;
    @DataField
    private final NodeType nodeType;
    @DataField
    private final NodeType parentType;
    @DataField
    private final NodeType parentOfParentType;
    @DataField
    private final NodeType oldParentType;
    @DataField
    private final NodeType oldParentOfParentType;
    @DataField
    private final String label;
    @DataField
    private final String oldLabel;

    public CodeChange() {
        this(0, 0, null, null, null,
                null, null, null,
                null, null);
    }

    private CodeChange(int originSolutionId, int targetSolutionId,
                      ChangeType changeType, NodeType nodeType, NodeType parentType, NodeType parentOfParentType,
                      NodeType oldParentType, NodeType oldParentOfParentType,
                      String label, String oldLabel) {
        this.changeType = changeType;
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.oldParentType = oldParentType;
        this.oldParentOfParentType = oldParentOfParentType;
        this.label = label;
        this.oldLabel = oldLabel;
    }

    public ChangeType getChangeType() {
        return changeType;
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

    public static CodeChange createDeleteChange(int originSolutionId, int targetSolutionId, NodeType nodeType,
                                                NodeType parentType, NodeType parentOfParentType,
                                                String label) {
        return new CodeChange(originSolutionId, targetSolutionId,
                DELETE, nodeType, parentType, parentOfParentType, NONE, NONE,
                label, "");
    }

    public static CodeChange createInsertChange(int originSolutionId, int targetSolutionId, NodeType nodeType,
                                                NodeType parentType, NodeType parentOfParentType,
                                                String label) {
        return new CodeChange(originSolutionId, targetSolutionId,
                INSERT, nodeType, parentType, parentOfParentType, NONE, NONE,
                label, "");
    }

    public static CodeChange createMoveChange(int originSolutionId, int targetSolutionId, NodeType nodeType,
                                                NodeType parentType, NodeType parentOfParentType,
                                                NodeType oldParentType, NodeType oldParentOfParentType,
                                                String label) {
        return new CodeChange(originSolutionId, targetSolutionId,
                MOVE, nodeType, parentType, parentOfParentType, oldParentType, oldParentOfParentType,
                label, "");
    }

    public static CodeChange createUpdateChange(int originSolutionId, int targetSolutionId, NodeType nodeType,
                                                NodeType parentType, NodeType parentOfParentType,
                                                String label, String oldLabel) {
        return new CodeChange(originSolutionId, targetSolutionId,
                UPDATE, nodeType, parentType, parentOfParentType, NONE, NONE,
                label, oldLabel);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CodeChange change = (CodeChange) o;

        if (changeType != change.changeType) return false;
        if (nodeType != change.nodeType) return false;
        if (parentType != change.parentType) return false;
        if (parentOfParentType != change.parentOfParentType) return false;
        if (oldParentType != change.oldParentType) return false;
        if (oldParentOfParentType != change.oldParentOfParentType) return false;
        if (label != null ? !label.equals(change.label) : change.label != null) return false;
        return oldLabel != null ? oldLabel.equals(change.oldLabel) : change.oldLabel == null;
    }

    @Override
    public int hashCode() {
        int result = changeType != null ? changeType.hashCode() : 0;
        result = 31 * result + (nodeType != null ? nodeType.hashCode() : 0);
        result = 31 * result + (parentType != null ? parentType.hashCode() : 0);
        result = 31 * result + (parentOfParentType != null ? parentOfParentType.hashCode() : 0);
        result = 31 * result + (oldParentType != null ? oldParentType.hashCode() : 0);
        result = 31 * result + (oldParentOfParentType != null ? oldParentOfParentType.hashCode() : 0);
        result = 31 * result + (label != null ? label.hashCode() : 0);
        result = 31 * result + (oldLabel != null ? oldLabel.hashCode() : 0);
        return result;
    }
}
