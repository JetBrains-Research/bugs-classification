package org.ml_methods_group.common;

import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.HashExtractor;

public class Hashers {
    public static final HashExtractor<ChangeType> CHANGE_TYPE_HASH = HashExtractor.<ChangeType>builder()
            .append("CT")
            .hashComponent(ChangeType::ordinal)
            .build();
    public static final HashExtractor<NodeType> NODE_TYPE_HASH = HashExtractor.<NodeType>builder()
            .append("NT")
            .hashComponent(NodeType::ordinal)
            .build();
    public static final HashExtractor<CodeChange.NodeState> TYPE_ONLY_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("TOS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeState> LABEL_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("LNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(CodeChange.NodeState::getLabel)
            .build();
    public static final HashExtractor<CodeChange.NodeState> JAVA_TYPE_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("JNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(CodeChange.NodeState::getJavaType)
            .build();
    public static final HashExtractor<CodeChange.NodeState> FULL_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("FNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeState::getLabel)
            .append(",")
            .hashComponent(CodeChange.NodeState::getJavaType)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> weak = HashExtractor.<CodeChange.NodeContext>builder()
            .append("TOC")
            .hashComponent(CodeChange.NodeContext::getNode, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> javaTypes = HashExtractor.<CodeChange.NodeContext>builder()
            .append("JTC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> full = HashExtractor.<CodeChange.NodeContext>builder()
            .append("FCC")
            .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> extended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("ECC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> fullExtended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("FEC")
            .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> deepExtended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("DEC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, LABEL_NODE_STATE_HASH)
            .build();

    public static HashExtractor<CodeChange> getCodeChangeHasher(HashExtractor<CodeChange.NodeContext> hasher) {
        return HashExtractor.<CodeChange>builder()
                .append("CCE")
                .hashComponent(CodeChange::getChangeType, CHANGE_TYPE_HASH)
                .append(",")
                .hashComponent(CodeChange::getOriginalContext, hasher)
                .append(",")
                .hashComponent(CodeChange::getDestinationContext, hasher)
                .build();
    }
}
