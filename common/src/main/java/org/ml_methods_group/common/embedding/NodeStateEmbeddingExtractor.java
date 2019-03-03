package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

public class NodeStateEmbeddingExtractor implements EmbeddingExtractor<NodeState> {

    private final Embedding<NodeType, ?> nodeStateEmbedding;
    private final Embedding<String, ?> labelEmbedding;
    private final Embedding<String, ?> javaTypeEmbedding;
    private final double[] noneVector;

    public NodeStateEmbeddingExtractor(Embedding<NodeType, ?> nodeTypeEmbedding,
                                       Embedding<String, ?> labelEmbedding,
                                       Embedding<String, ?> javaTypeEmbedding) {
        this.nodeStateEmbedding = nodeTypeEmbedding;
        this.labelEmbedding = labelEmbedding;
        this.javaTypeEmbedding = javaTypeEmbedding;
        this.noneVector = getDefaultVector();
    }

    @Override
    public double[] process(NodeState value) {
        if (javaTypeEmbedding != null) {
            return FunctionsUtils.sum(
                    nodeStateEmbedding.vectorFor(value.getType()),
                    labelEmbedding.vectorFor(value.getLabel()),
                    javaTypeEmbedding.vectorFor(value.getJavaType()));
        } else {
            return FunctionsUtils.sum(
                    nodeStateEmbedding.vectorFor(value.getType()),
                    labelEmbedding.vectorFor(value.getLabel()));
        }
    }

    @Override
    public double[] defaultVector() {
        return noneVector;
    }

    private double[] getDefaultVector() {
        if (javaTypeEmbedding != null) {
            return FunctionsUtils.sum(
                    nodeStateEmbedding.defaultVector(),
                    labelEmbedding.defaultVector(),
                    javaTypeEmbedding.defaultVector());
        } else {
            return FunctionsUtils.sum(
                    nodeStateEmbedding.defaultVector(),
                    labelEmbedding.defaultVector());
        }
    }
}
