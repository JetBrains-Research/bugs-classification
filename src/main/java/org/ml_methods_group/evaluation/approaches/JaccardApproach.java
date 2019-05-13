package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.extractors.HashExtractor;
import org.ml_methods_group.common.extractors.PointwiseExtractor;
import org.ml_methods_group.common.metrics.functions.JaccardDistanceFunction;

import java.util.List;

import static org.ml_methods_group.evaluation.approaches.BOWApproach.*;

public class JaccardApproach {
    public static Approach<List<String>> getDefaultApproach(FeaturesExtractor<Solution, Changes> generator) {
        final HashExtractor<NodeContext> hasher = HashExtractor.<NodeContext>builder()
                .append("FTCC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<CodeChange> extractor = getCodeChangeHasher(hasher);
        return new Approach<>(generator.compose(Changes::getChanges)
                .compose(new PointwiseExtractor<>(extractor)),
                new JaccardDistanceFunction<>(), "def_jac");
    }

    public static Approach<List<String>> getExtendedApproach(FeaturesExtractor<Solution, Changes> generator) {
        final HashExtractor<NodeContext> hasher = HashExtractor.<NodeContext>builder()
                .append("FECC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<CodeChange> extractor = getCodeChangeHasher(hasher);
        return new Approach<>(generator.compose(Changes::getChanges)
                .compose(new PointwiseExtractor<>(extractor)),
                new JaccardDistanceFunction<>(), "ext_jac");
    }

    public static Approach<List<String>> getFullApproach(FeaturesExtractor<Solution, Changes> generator) {
        final HashExtractor<NodeContext> hasher = HashExtractor.<NodeContext>builder()
                .append("FFCC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, FULL_NODE_STATE_HASH)
                .build();
        final HashExtractor<CodeChange> extractor = getCodeChangeHasher(hasher);
        return new Approach<>(generator.compose(Changes::getChanges)
                .compose(new PointwiseExtractor<>(extractor)),
                new JaccardDistanceFunction<>(), "ful_jac");
    }

    public static final ApproachTemplate<List<String>> TEMPLATE = (d, g) -> getDefaultApproach(g);
}
