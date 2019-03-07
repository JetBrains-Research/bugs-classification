package org.ml_methods_group.common.ast.changes;

import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.ast.NodeType;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import static org.ml_methods_group.common.ast.changes.CodeChange.NodeState.getState;
import static org.ml_methods_group.common.ast.changes.CodeChange.NodeState.getStates;

public class CodeChange implements Serializable {

    public static final String NO_LABEL = "NO_LABEL";
    private static final NodeState NONE_STATE = new NodeState(NodeType.NONE, NO_LABEL, NO_LABEL, NO_LABEL, 0);
    private static final NodeState[] EMPTY_STATE_ARRAY = new NodeState[0];

    private final NodeContext originalContext;
    private final NodeContext destinationContext;
    private final ChangeType changeType;

    public CodeChange(NodeContext originalContext, NodeContext destinationContext, ChangeType changeType) {
        this.originalContext = originalContext;
        this.destinationContext = destinationContext;
        this.changeType = changeType;
    }

    public NodeContext getOriginalContext() {
        return originalContext;
    }

    public NodeContext getDestinationContext() {
        return destinationContext;
    }

    public ChangeType getChangeType() {
        return changeType;
    }

    public static CodeChange fromAction(Action action, MappingStore mappings) {
        if (action.getClass() == Insert.class) {
            return fromAction((Insert) action, mappings);
        } else if (action.getClass() == Delete.class) {
            return fromAction((Delete) action, mappings);
        } else if (action.getClass() == Move.class) {
            return fromAction((Move) action, mappings);
        } else {
            return fromAction((Update) action, mappings);
        }
    }

    private static NodeContext emptyContext(ITree node) {
        return new NodeContext(getState(node), NONE_STATE, NONE_STATE, EMPTY_STATE_ARRAY,
                EMPTY_STATE_ARRAY, EMPTY_STATE_ARRAY);
    }

    private static CodeChange fromAction(Insert insert, MappingStore mappings) {
        final ITree node = insert.getNode();
        return new CodeChange(emptyContext(node), NodeContext.getContext(node), ChangeType.INSERT);
    }

    private static CodeChange fromAction(Delete delete, MappingStore mappings) {
        final ITree node = delete.getNode();
        return new CodeChange(NodeContext.getContext(node), emptyContext(node), ChangeType.DELETE);
    }

    private static CodeChange fromAction(Move move, MappingStore mappings) {
        final ITree originalNode = move.getNode();
        final ITree destinationNode = mappings.getDst(originalNode);
        return new CodeChange(NodeContext.getContext(originalNode),
                NodeContext.getContext(destinationNode),
                ChangeType.MOVE);
    }

    private static CodeChange fromAction(Update update, MappingStore mappings) {
        final ITree originalNode = update.getNode();
        final ITree destinationNode = mappings.getDst(originalNode);
        return new CodeChange(NodeContext.getContext(originalNode),
                NodeContext.getContext(destinationNode),
                ChangeType.UPDATE);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CodeChange that = (CodeChange) o;

        if (!originalContext.equals(that.originalContext)) return false;
        if (!destinationContext.equals(that.destinationContext)) return false;
        return changeType == that.changeType;
    }

    @Override
    public int hashCode() {
        int result = originalContext.hashCode();
        result = 31 * result + destinationContext.hashCode();
        result = 31 * result + changeType.hashCode();
        return result;
    }

    public static class NodeState implements Serializable {

        private final NodeType type;
        private final String javaType;
        private final String label;
        private final String originalLabel;
        private final int positionInParent;

        public NodeState(NodeType type, String javaType, String label, String originalLabel, int positionInParent) {
            this.type = type;
            this.javaType = javaType == null ? NO_LABEL : javaType;
            this.label = label.isEmpty() ? NO_LABEL : label;
            this.originalLabel = originalLabel == null ? label : originalLabel;
            this.positionInParent = positionInParent;
        }

        public static NodeState getState(ITree node) {
            if (node == null || node.getType() == -1) {
                return NONE_STATE;
            }
            final NodeType type = NodeType.valueOf(node.getType());
            final String javaType = Optional.ofNullable(node.getMetadata(MetadataKeys.JAVA_TYPE))
                    .map(Object::toString)
                    .orElse(NO_LABEL);
            final String label = node.getLabel().isEmpty() ? NO_LABEL : node.getLabel();
            final String originalLabel = Optional.ofNullable(node.getMetadata(MetadataKeys.ORIGINAL_NAME))
                    .map(Object::toString)
                    .orElse(label);
            final int positionInParent = node.positionInParent();
            return new NodeState(type, javaType, label, originalLabel, positionInParent);
        }

        public static NodeState[] getStates(List<ITree> nodes) {
            if (nodes == null || nodes.isEmpty()) {
                return EMPTY_STATE_ARRAY;
            }
            return nodes.stream()
                    .map(NodeState::getState)
                    .toArray(NodeState[]::new);
        }

        public NodeType getType() {
            return type;
        }

        public String getJavaType() {
            return javaType;
        }

        public String getOriginalLabel() {
            return originalLabel;
        }

        public String getLabel() {
            return label;
        }

        public int getPositionInParent() {
            return positionInParent;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            NodeState nodeState = (NodeState) o;

            if (positionInParent != nodeState.positionInParent) return false;
            if (type != nodeState.type) return false;
            if (!Objects.equals(javaType, nodeState.javaType)) return false;
            if (!label.equals(nodeState.label)) return false;
            return Objects.equals(originalLabel, nodeState.originalLabel);
        }

        @Override
        public int hashCode() {
            int result = type.hashCode();
            result = 31 * result + (javaType != null ? javaType.hashCode() : 0);
            result = 31 * result + label.hashCode();
            result = 31 * result + (originalLabel != null ? originalLabel.hashCode() : 0);
            result = 31 * result + positionInParent;
            return result;
        }

        @Override
        public String toString() {
            return "Node{" + type + ": " + label + "}";
        }
    }

    public static class NodeContext implements Serializable {

        private final NodeState node;
        private final NodeState parent;
        private final NodeState parentOfParent;
        private final NodeState[] uncles;
        private final NodeState[] brothers;
        private final NodeState[] children;


        public NodeContext(NodeState node, NodeState parent, NodeState parentOfParent, NodeState[] uncles,
                           NodeState[] brothers, NodeState[] children) {
            this.node = node;
            this.parent = parent;
            this.parentOfParent = parentOfParent;
            this.uncles = uncles;
            this.brothers = brothers;
            this.children = children;
        }

        public static NodeContext getContext(ITree node) {
            final ITree parent = node.getParent();
            final ITree parentOfParent = parent == null ? null : parent.getParent();
            return new NodeContext(NodeState.getState(node),
                    NodeState.getState(parent),
                    NodeState.getState(parentOfParent),
                    parentOfParent == null ? EMPTY_STATE_ARRAY : getStates(parentOfParent.getChildren()),
                    parent == null ? EMPTY_STATE_ARRAY : getStates(parent.getChildren()),
                    getStates(node.getChildren()));
        }

        public NodeState getNode() {
            return node;
        }

        public NodeState getParent() {
            return parent;
        }

        public NodeState getParentOfParent() {
            return parentOfParent;
        }

        public NodeState[] getUncles() {
            return uncles;
        }

        public NodeState[] getBrothers() {
            return brothers;
        }

        public NodeState[] getChildren() {
            return children;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            NodeContext that = (NodeContext) o;

            if (!node.equals(that.node)) return false;
            if (!parent.equals(that.parent)) return false;
            if (!parentOfParent.equals(that.parentOfParent)) return false;
            // Probably incorrect - comparing Object[] arrays with Arrays.equals
            if (!Arrays.equals(uncles, that.uncles)) return false;
            // Probably incorrect - comparing Object[] arrays with Arrays.equals
            if (!Arrays.equals(brothers, that.brothers)) return false;
            // Probably incorrect - comparing Object[] arrays with Arrays.equals
            return Arrays.equals(children, that.children);
        }

        @Override
        public int hashCode() {
            int result = node.hashCode();
            result = 31 * result + parent.hashCode();
            result = 31 * result + parentOfParent.hashCode();
            result = 31 * result + Arrays.hashCode(uncles);
            result = 31 * result + Arrays.hashCode(brothers);
            result = 31 * result + Arrays.hashCode(children);
            return result;
        }
    }
}
