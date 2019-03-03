package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.embedding.SequenceContextEmbeddingExtractor.SequenceContext;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

public class NodeContextEmbeddingExtractor implements EmbeddingExtractor<NodeContext> {

    private final EmbeddingExtractor<NodeState> nodeEmbedding;
    private final EmbeddingExtractor<NodeState> parentEmbedding;
    private final EmbeddingExtractor<NodeState> parentOfParentEmbedding;
    private final PrefixEmbeddingExtractor<NodeState> childrenEmbedding;
    private final SequenceContextEmbeddingExtractor<NodeState> brothersEmbedding;
    private final SequenceContextEmbeddingExtractor<NodeState> unclesEmbedding;

    public NodeContextEmbeddingExtractor(EmbeddingExtractor<NodeState> nodeEmbedding,
                                         EmbeddingExtractor<NodeState> parentEmbedding,
                                         EmbeddingExtractor<NodeState> parentOfParentEmbedding,
                                         PrefixEmbeddingExtractor<NodeState> childrenEmbedding,
                                         SequenceContextEmbeddingExtractor<NodeState> brothersEmbedding,
                                         SequenceContextEmbeddingExtractor<NodeState> unclesEmbedding) {
        this.nodeEmbedding = nodeEmbedding;
        this.parentEmbedding = parentEmbedding;
        this.parentOfParentEmbedding = parentOfParentEmbedding;
        this.childrenEmbedding = childrenEmbedding;
        this.brothersEmbedding = brothersEmbedding;
        this.unclesEmbedding = unclesEmbedding;
    }

    @Override
    public double[] process(NodeContext value) {
        return FunctionsUtils.sum(
                nodeEmbedding.process(value.getNode()),
                parentEmbedding.process(value.getParent()),
                parentOfParentEmbedding.process(value.getParentOfParent()),
                unclesEmbedding.process(new SequenceContext<>(value.getUncles(),
                        value.getNode().getPositionInParent())),
                brothersEmbedding.process(new SequenceContext<>(value.getBrothers(),
                        value.getNode().getPositionInParent())),
                childrenEmbedding.process(value.getChildren())
        );
    }

    @Override
    public double[] defaultVector() {
        return FunctionsUtils.sum(
                nodeEmbedding.defaultVector(),
                parentEmbedding.defaultVector(),
                parentOfParentEmbedding.defaultVector(),
                unclesEmbedding.defaultVector(),
                brothersEmbedding.defaultVector(),
                childrenEmbedding.defaultVector()
        );
    }
}
