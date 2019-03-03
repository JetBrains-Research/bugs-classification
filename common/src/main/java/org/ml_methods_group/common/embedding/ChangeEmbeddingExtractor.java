package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

public class ChangeEmbeddingExtractor implements EmbeddingExtractor<CodeChange> {

    private final Embedding<ChangeType, ?> changeTypeEmbedding;
    private final EmbeddingExtractor<NodeContext> originalContextEmbedding;
    private final EmbeddingExtractor<NodeContext> destinationContextEmbedding;

    public ChangeEmbeddingExtractor(Embedding<ChangeType, ?> changeTypeEmbedding,
                                    EmbeddingExtractor<NodeContext> originalContextEmbedding,
                                    EmbeddingExtractor<NodeContext> destinationContextEmbedding) {
        this.changeTypeEmbedding = changeTypeEmbedding;
        this.originalContextEmbedding = originalContextEmbedding;
        this.destinationContextEmbedding = destinationContextEmbedding;
    }

    @Override
    public double[] process(CodeChange value) {
        return FunctionsUtils.sum(
                changeTypeEmbedding.vectorFor(value.getChangeType()),
                originalContextEmbedding.process(value.getOriginalContext()),
                destinationContextEmbedding.process(value.getDestinationContext())
        );
    }

    @Override
    public double[] defaultVector() {
        return FunctionsUtils.sum(
                changeTypeEmbedding.defaultVector(),
                originalContextEmbedding.defaultVector(),
                destinationContextEmbedding.defaultVector()
        );
    }
}
