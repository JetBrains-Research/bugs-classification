package org.ml_methods_group.common.changes;

import java.util.Arrays;

public class UpdateChange implements CodeChange {

    private final NodeState node;
    private final String oldLabel;
    private final NodeState parent;
    private final NodeState parentOfParent;
    private final NodeState[] children;
    private final NodeState[] brothers;
    private final int hash;

    public UpdateChange(NodeState node, String oldLabel, NodeState parent, NodeState parentOfParent,
                        NodeState[] children, NodeState[] brothers) {
        this.node = node;
        this.oldLabel = oldLabel;
        this.parent = parent;
        this.parentOfParent = parentOfParent;
        this.children = children;
        this.brothers = brothers;

        int hash = node.hashCode();
        hash = 31 * hash + oldLabel.hashCode();
        hash = 31 * hash + parent.hashCode();
        hash = 31 * hash + parentOfParent.hashCode();
        hash = 31 * hash + Arrays.hashCode(children);
        hash = 31 * hash + Arrays.hashCode(brothers);

        this.hash = hash;
    }

    @Override
    public NodeState getNode() {
        return node;
    }

    public String getOldLabel() {
        return oldLabel;
    }

    @Override
    public NodeState getParent() {
        return parent;
    }

    @Override
    public NodeState getParentOfParent() {
        return parentOfParent;
    }

    @Override
    public NodeState[] getChildren() {
        return children;
    }

    @Override
    public NodeState[] getBrothers() {
        return brothers;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.UPDATE;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (hashCode() != o.hashCode()) return false;

        UpdateChange that = (UpdateChange) o;

        if (!node.equals(that.node)) return false;
        if (!oldLabel.equals(that.oldLabel)) return false;
        if (!parent.equals(that.parent)) return false;
        if (!parentOfParent.equals(that.parentOfParent)) return false;
        return Arrays.equals(children, that.children) && Arrays.equals(brothers, that.brothers);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    public String toString() {
        return "UpdateChange{" + System.lineSeparator() +
                "\tnode=" + node + System.lineSeparator() +
                "\toldLabel=" + oldLabel + System.lineSeparator() +
                "\tparent=" + parent + System.lineSeparator() +
                "\tparentOfParent=" + parentOfParent + System.lineSeparator() +
                "\tchildren=" + Arrays.toString(children) + System.lineSeparator() +
                "\tbrothers=" + Arrays.toString(brothers) + System.lineSeparator() +
                '}';
    }
}
