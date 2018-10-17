package org.ml_methods_group.common.changes;

import java.util.Arrays;

public class DeleteChange implements CodeChange {
    private final NodeState node;
    private final NodeState parent;
    private final NodeState parentOfParent;
    private final NodeState[] children;
    private final NodeState[] brothers;
    private final int hash;

    public DeleteChange(NodeState node, NodeState parent, NodeState parentOfParent,
                        NodeState[] children,
                        NodeState[] brothers) {
        this.node = node;
        this.parent = parent;
        this.parentOfParent = parentOfParent;
        this.children = children;
        this.brothers = brothers;

        int hash = node.hashCode();
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

    @Override
    public NodeState[] getChildren() {
        return children;
    }

    @Override
    public NodeState[] getBrothers() {
        return brothers;
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
    public ChangeType getChangeType() {
        return ChangeType.DELETE;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (hashCode() != o.hashCode()) return false;

        DeleteChange that = (DeleteChange) o;

        if (!node.equals(that.node)) return false;
        if (!parent.equals(that.parent)) return false;
        if (!parentOfParent.equals(that.parentOfParent)) return false;
        if (!Arrays.equals(children, that.children)) return false;
        return Arrays.equals(brothers, that.brothers);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    public String toString() {
        return "DeleteChange{" + System.lineSeparator() +
                "\tnode=" + node + System.lineSeparator() +
                "\tparent=" + parent + System.lineSeparator() +
                "\tparentOfParent=" + parentOfParent + System.lineSeparator() +
                "\tchildren=" + Arrays.toString(children) + System.lineSeparator() +
                "\tbrothers=" + Arrays.toString(brothers) + System.lineSeparator() +
                '}';
    }
}
