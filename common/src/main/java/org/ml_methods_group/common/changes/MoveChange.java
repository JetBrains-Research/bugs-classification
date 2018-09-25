package org.ml_methods_group.common.changes;


import java.util.Arrays;

public class MoveChange implements CodeChange {
    private final NodeState node;
    private final NodeState parent;
    private final NodeState oldParent;
    private final NodeState parentOfParent;
    private final NodeState oldParentOfParent;
    private final NodeState[] children;
    private final NodeState[] brothers;
    private final NodeState[] oldBrothers;
    private final int hash;

    public MoveChange(NodeState node, NodeState parent, NodeState oldParent,
                      NodeState parentOfParent, NodeState oldParentOfParent,
                      NodeState[] children, NodeState[] brothers, NodeState[] oldBrothers) {
        this.node = node;
        this.parent = parent;
        this.oldParent = oldParent;
        this.parentOfParent = parentOfParent;
        this.oldParentOfParent = oldParentOfParent;
        this.children = children;
        this.brothers = brothers;
        this.oldBrothers = oldBrothers;

        int hash = node.hashCode();
        hash = 31 * hash + parent.hashCode();
        hash = 31 * hash + oldParent.hashCode();
        hash = 31 * hash + parentOfParent.hashCode();
        hash = 31 * hash + oldParentOfParent.hashCode();
        hash = 31 * hash + Arrays.hashCode(children);
        hash = 31 * hash + Arrays.hashCode(brothers);
        hash = 31 * hash + Arrays.hashCode(oldBrothers);

        this.hash = hash;
    }

    @Override
    public NodeState getNode() {
        return node;
    }

    @Override
    public NodeState getParent() {
        return parent;
    }

    public NodeState getOldParent() {
        return oldParent;
    }

    @Override
    public NodeState getParentOfParent() {
        return parentOfParent;
    }

    public NodeState getOldParentOfParent() {
        return oldParentOfParent;
    }

    @Override
    public NodeState[] getChildren() {
        return children;
    }

    @Override
    public NodeState[] getBrothers() {
        return brothers;
    }

    public NodeState[] getOldBrothers() {
        return oldBrothers;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.MOVE;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (hashCode() != o.hashCode()) return false;

        MoveChange that = (MoveChange) o;

        if (!node.equals(that.node)) return false;
        if (!parent.equals(that.parent)) return false;
        if (!oldParent.equals(that.oldParent)) return false;
        if (!parentOfParent.equals(that.parentOfParent)) return false;
        if (!oldParentOfParent.equals(that.oldParentOfParent)) return false;
        if (!Arrays.equals(children, that.children)) return false;
        return Arrays.equals(brothers, that.brothers) && Arrays.equals(oldBrothers, that.oldBrothers);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    public String toString() {
        return "MoveChange{" + System.lineSeparator() +
                "\tnode=" + node + System.lineSeparator() +
                "\tparent=" + parent + System.lineSeparator() +
                "\toldParent=" + oldParent + System.lineSeparator() +
                "\tparentOfParent=" + parentOfParent + System.lineSeparator() +
                "\toldParentOfParent=" + oldParentOfParent + System.lineSeparator() +
                "\tchildren=" + Arrays.toString(children) + System.lineSeparator() +
                "\tbrothers=" + Arrays.toString(brothers) + System.lineSeparator() +
                "\toldBrothers=" + Arrays.toString(oldBrothers) + System.lineSeparator() +
                '}';
    }
}
