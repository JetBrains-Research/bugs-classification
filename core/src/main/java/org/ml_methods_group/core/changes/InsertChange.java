package org.ml_methods_group.core.changes;

import java.util.Arrays;

public class InsertChange implements CodeChange {
    private final CodeChange.NodeState node;
    private final CodeChange.NodeState parent;
    private final CodeChange.NodeState parentOfParent;
    private final CodeChange.NodeState[] children;
    private final CodeChange.NodeState[] brothers;
    private final int hash;

    public InsertChange(CodeChange.NodeState node, CodeChange.NodeState parent, CodeChange.NodeState parentOfParent,
                        CodeChange.NodeState[] children,
                        CodeChange.NodeState[] brothers) {
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
    public CodeChange.NodeState getNode() {
        return node;
    }

    @Override
    public CodeChange.NodeState[] getChildren() {
        return children;
    }

    @Override
    public CodeChange.NodeState[] getBrothers() {
        return brothers;
    }

    @Override
    public CodeChange.NodeState getParent() {
        return parent;
    }

    @Override
    public CodeChange.NodeState getParentOfParent() {
        return parentOfParent;
    }

    @Override
    public ChangeType getChangeType() {
        return ChangeType.INSERT;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (hashCode() != o.hashCode()) return false;

        InsertChange that = (InsertChange) o;

        if (hash != that.hash) return false;
        if (!node.equals(that.node)) return false;
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
        return "InsertChange{" + System.lineSeparator() +
                "\tnode=" + node + System.lineSeparator() +
                "\tparent=" + parent + System.lineSeparator() +
                "\tparentOfParent=" + parentOfParent + System.lineSeparator() +
                "\tchildren=" + Arrays.toString(children) + System.lineSeparator() +
                "\tbrothers=" + Arrays.toString(brothers) + System.lineSeparator() +
                '}';
    }
}
