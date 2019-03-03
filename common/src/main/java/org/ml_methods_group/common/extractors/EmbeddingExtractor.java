//package org.ml_methods_group.common.extractors;
//
//import org.ml_methods_group.common.FeaturesExtractor;
//import org.ml_methods_group.common.ast.*;
//import org.ml_methods_group.common.ast.changes.ChangeType;
//import org.ml_methods_group.common.ast.changes.CodeChange;
//import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
//import org.ml_methods_group.common.embedding.Embedding;
//
//import java.io.FileNotFoundException;
//
//public class EmbeddingExtractor implements FeaturesExtractor<CodeChange, EmbeddingExtractor.EmbeddedChange> {
//
//    private final Embedding<NodeType> nodes;
//    private final Embedding<String> labels;
//
//    public EmbeddingExtractor(Embedding<NodeType> nodes, Embedding<String> labels) throws FileNotFoundException {
//        this.nodes = nodes;
//        this.labels = labels;
//    }
//
//    @Override
//    public EmbeddedChange process(CodeChange value) {
//        final double[] result = new double[8 * nodes.size() + 9 * labels.size()];
//        int index = 0;
//        index = fill(result, index, 4, value.getNode());
//        index = fill(result, index, 2, value.getParent());
//        index = fill(result, index, 2, value.getParentOfParent());
//        index = fill(result, index, 1, value.getBrothers());
//        index = fill(result, index, 1, value.getChildren());
//        if (value instanceof MoveChange) {
//            final MoveChange move = (MoveChange) value;
//            index = fill(result, index, 1.5, move.getOldParent());
//            index = fill(result, index, 1.5, move.getOldParentOfParent());
//            index = fill(result, index, 1, move.getOldBrothers());
//        } else {
//            index = fill(result, index, 1.5, value.getParent());
//            index = fill(result, index, 1.5, value.getParentOfParent());
//            index = fill(result, index, 1, value.getBrothers());
//        }
//        if (value instanceof UpdateChange) {
//            final UpdateChange update = (UpdateChange) value;
//            index = fill(result, index, 2, update.getOldLabel());
//        } else {
//            index = fill(result, index, 2, value.getLabel());
//        }
//        assert index == result.length;
//        return new EmbeddedChange(value.getChangeType(), result);
//    }
//
//    private int fill(double[] result, int shift, double modifier, String label) {
//        add(result, shift, labels.vectorFor(label));
//        scale(result, shift, labels.size(), modifier);
//        return shift + labels.size();
//    }
//
//    private int fill(double[] result, int shift, double modifier, NodeState... states) {
//        for (NodeState state : states) {
//            final double[] node = nodes.vectorFor(state.getType());
//            final double[] label = labels.vectorFor(state.getLabel());
//            add(result, shift, node);
//            add(result, shift + node.length, label);
//        }
//        if (states.length != 0) {
//            scale(result, shift, nodes.size() + labels.size(), modifier / states.length);
//        }
//        return shift + nodes.size() + labels.size();
//    }
//
//    private void add(double[] dst, int shift, double[] src) {
//        for (int i = 0; i < src.length; i++) {
//            dst[i + shift] += src[i];
//        }
//    }
//
//    private void scale(double[] dst, int shift, int length, double value) {
//        for (int i = 0; i < length; i++) {
//            dst[i + shift] *= value;
//        }
//    }
//
//    public static class EmbeddedChange {
//        private final ChangeType type;
//        private final double[] vector;
//
//        public EmbeddedChange(ChangeType type, double[] vector) {
//            this.type = type;
//            this.vector = vector;
//        }
//
//        public ChangeType getType() {
//            return type;
//        }
//
//        public double[] getVector() {
//            return vector;
//        }
//    }
//}
