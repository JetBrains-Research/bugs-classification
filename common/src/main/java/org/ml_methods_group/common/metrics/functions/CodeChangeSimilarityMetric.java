package org.ml_methods_group.common.metrics.functions;

import org.ml_methods_group.common.SimilarityMetric;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.metrics.algorithms.DamerauLevenshteinDistance;

import static org.ml_methods_group.common.ast.changes.CodeChange.NO_LABEL;

public class CodeChangeSimilarityMetric implements SimilarityMetric<CodeChange> {
    @Override
    public double measure(CodeChange first, CodeChange second) {
        if (getElementType(first) != getElementType(second)) {
            return 0;
        }
        switch (first.getChangeType()) {
            case MOVE:
                return matchMove(first, second);
            case DELETE:
                return matchDelete(first, second);
            case UPDATE:
                return matchUpdate(first, second);
            case INSERT:
                return matchInsert(first, second);
            default:
                throw new RuntimeException("Unexpected enum type: " + first.getChangeType());
        }
    }

    private double matchInsert(CodeChange a, CodeChange b) {
        return matchState(a.getDestinationContext().getNode(), b.getDestinationContext().getNode()) *
                matchParents(a.getDestinationContext(), b.getDestinationContext());
    }

    private double matchDelete(CodeChange a, CodeChange b) {
        return matchState(a.getOriginalContext().getNode(), b.getOriginalContext().getNode()) *
                matchParents(a.getOriginalContext(), b.getOriginalContext());
    }

    private double matchUpdate(CodeChange a, CodeChange b) {
        return mean(matchState(a.getOriginalContext().getNode(), b.getOriginalContext().getNode()),
                matchState(a.getDestinationContext().getNode(), b.getDestinationContext().getNode())) *
                mean(matchParents(a.getDestinationContext(), b.getDestinationContext()),
                        matchParents(a.getOriginalContext(), b.getOriginalContext()));
    }

    private double matchMove(CodeChange a, CodeChange b) {
        return mean(matchState(a.getOriginalContext().getNode(), b.getOriginalContext().getNode()),
                matchState(a.getDestinationContext().getNode(), b.getDestinationContext().getNode())) *
                mean(matchParents(a.getDestinationContext(), b.getDestinationContext()),
                        matchParents(a.getOriginalContext(), b.getOriginalContext()));
    }

    private double matchState(NodeState a, NodeState b) {
        if (a.getType() != b.getType()) {
            return 0;
        }
        final double labelModifier = a.getLabel().equals(b.getLabel()) ? 1 : 0.5;
        final double typeModifier = a.getJavaType().equals(b.getJavaType()) ? 1 : 0.5;
        return !a.getJavaType().equals(NO_LABEL) || !b.getJavaType().equals(NO_LABEL) ?
                (labelModifier + typeModifier) / 2 : labelModifier;
    }

    private double matchLabels(String a, String b) {
        return DamerauLevenshteinDistance.problemFor(a, b).solve();
    }

    private double matchParents(NodeContext a, NodeContext b) {
        return matchState(a.getParent(), b.getParent()) * matchState(a.getParentOfParent(), b.getParentOfParent());
    }

    private double matchTypes(NodeState[] a, NodeState[] b) {
        if (a.length == 0 && b.length == 0) {
            return 1;
        }
        final int penalty = DamerauLevenshteinDistance.problemFor(a, b, (x, y) -> x.getType() == y.getType()).solve();
        return 1 - (double) penalty / Math.max(a.length, b.length);
    }

    private double mean(double a, double b) {
        return (a + b) / 2;
    }

    @Override
    public int getElementType(CodeChange value) {
        return value.getChangeType().ordinal() + 10 * value.getOriginalContext().getNode().getType().ordinal() +
                10000 * value.getDestinationContext().getNode().getType().ordinal();
    }
}
