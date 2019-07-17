package org.ml_methods_group.common.extractors;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.ast.NodeType;

import java.util.HashMap;
import java.util.Map;

public class HeuristicASTRepresentationExtractor implements FeaturesExtractor<ITree,
        HeuristicASTRepresentationExtractor.HeuristicASTRepresentation> {

    private final Map<Integer, Integer> indexes = new HashMap<>();

    @Override
    public synchronized HeuristicASTRepresentation process(ITree value) {
        final int[] counters = new int[NodeType.values().length];
        int maxLimit = 0;
        int sum = 0;
        for (final var node : value.preOrder()) {
            final int index = indexes.computeIfAbsent(node.getType(), x -> indexes.size());
            maxLimit = Math.max(index, maxLimit);
            counters[index]++;
            sum++;
        }
        if (sum != value.getSize()) {
            throw new RuntimeException();
        }
        return new HeuristicASTRepresentation(counters, maxLimit, value.getSize());
    }

    public DistanceFunction<HeuristicASTRepresentation> getDistanceFunction() {
        return new HeuristicDistance();
    }

    private class HeuristicDistance implements DistanceFunction<HeuristicASTRepresentation> {

        @Override
        public double distance(HeuristicASTRepresentation first, HeuristicASTRepresentation second) {
            return distance(first, second, Double.POSITIVE_INFINITY);
        }

        @Override
        public double distance(HeuristicASTRepresentation first, HeuristicASTRepresentation second, double upperBound) {
            if (first.getSource() != second.getSource()) {
                throw new IllegalStateException();
            }
            if (Math.abs(first.sum - second.sum) >= upperBound) {
                return upperBound;
            }
            final int limit = Math.max(first.limit, second.limit);
            int diff = 0;
            for (int i = 0; i < limit && diff < upperBound; i++) {
                final int delta = first.counters[i] - second.counters[i];
                if (delta > 0) {
                    diff += delta;
                } else {
                    diff -= delta;
                }
            }
            return diff >= upperBound ? upperBound : diff;
        }
    }

    public class HeuristicASTRepresentation {
        private final int[] counters;
        private final int limit;
        private final int sum;

        private HeuristicASTRepresentation(int[] counters, int limit, int sum) {
            this.counters = counters;
            this.limit = limit;
            this.sum = sum;
        }

        private HeuristicASTRepresentationExtractor getSource() {
            return HeuristicASTRepresentationExtractor.this;
        }
    }
}
