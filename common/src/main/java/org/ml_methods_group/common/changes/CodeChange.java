package org.ml_methods_group.common.changes;

import java.io.Serializable;

public interface CodeChange extends Serializable {

    String NO_LABEL = "NO_LABEL";
    NodeState NONE_STATE = new NodeState(NodeType.NONE, NO_LABEL);
    NodeState MY_STATE = new NodeState(NodeType.NONE, "MY_STATE_NODE");
    NodeState[] EMPTY_STATE_ARRAY = new NodeState[0];

    default String getLabel() {
        return getNode().label;
    }

    default NodeType getNodeType() {
        return getNode().type;
    }

    NodeState getNode();

    NodeState[] getChildren();

    NodeState[] getBrothers();

    NodeState getParent();

    NodeState getParentOfParent();

    ChangeType getChangeType();


    class NodeState implements Serializable {

        private final NodeType type;
        private final String label;

        public NodeState(NodeType type, String label) {
            this.type = type;
            this.label = label.isEmpty() ? NO_LABEL : label;
        }

        public NodeType getType() {
            return type;
        }

        public String getLabel() {
            return label;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            NodeState nodeState = (NodeState) o;

            return type == nodeState.type && label.equals(nodeState.label);
        }

        @Override
        public int hashCode() {
            return type.hashCode() * 31 + label.hashCode();
        }

        @Override
        public String toString() {
            return "Node{" + type + ": " + label + "}";
        }
    }
}
