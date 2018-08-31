package org.ml_methods_group.core.basic.metrics;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.algorithms.DamerauLevenshteinDistance;
import org.ml_methods_group.core.changes.*;
import org.ml_methods_group.core.changes.CodeChange.NodeState;

import java.util.Arrays;
import java.util.function.Function;

import static org.ml_methods_group.core.changes.NodeType.INFIX_EXPRESSION;
import static org.ml_methods_group.core.changes.NodeType.METHOD_INVOCATION;
import static org.ml_methods_group.core.changes.NodeType.STRING_LITERAL;

public class ChangeDistanceFunction implements DistanceFunction<CodeChange> {

    private final ComparisionProperties properties;

    public ChangeDistanceFunction(ComparisionProperties properties) {
        this.properties = properties;
    }

    public ChangeDistanceFunction() {
        this(defaultProperties());
    }

    private double matchInsertChanges(InsertChange first, InsertChange second) {
        final NodeType type = first.getNodeType();
        return matchLabel(first.getLabel(), second.getLabel(),
                properties.getLabelImportance(type), 4, 0.3, 0.1)
                * matchStates(first, second, CodeChange::getParent, properties::getAsParentImportance)
                * matchStates(first, second, CodeChange::getParentOfParent, properties::getAsParentOfParentImportance)
                * matchArrays(first.getChildren(), second.getChildren(),
                properties.getChildrenImportance(type), 0.1);
    }

    private double matchDeleteChanges(DeleteChange first, DeleteChange second) {
        final NodeType type = first.getNodeType();
        return matchLabel(first.getLabel(), second.getLabel(),
                properties.getLabelImportance(type), 4, 0.3, 0.1)
                * matchStates(first, second, CodeChange::getParent, properties::getAsParentImportance)
                * matchStates(first, second, CodeChange::getParentOfParent, properties::getAsParentOfParentImportance)
                * matchArrays(first.getChildren(), second.getChildren(),
                properties.getChildrenImportance(type), 0.1);
    }

    private double matchUpdateChanges(UpdateChange first, UpdateChange second) {
        final NodeType type = first.getNodeType();
        final double labelImportance = properties.getLabelImportance(type);
        return matchLabel(first.getLabel(), second.getLabel(), labelImportance, 4, 0.3, 0.1)
                * matchLabel(first.getOldLabel(), second.getOldLabel(), labelImportance, 4, 0.3, 0.1)
                * matchStates(first, second, CodeChange::getParent, properties::getAsParentImportance)
                * matchStates(first, second, CodeChange::getParentOfParent, properties::getAsParentOfParentImportance)
                * matchArrays(first.getChildren(), second.getChildren(),
                properties.getChildrenImportance(type), 0.1);
    }

    private double matchMoveChanges(MoveChange first, MoveChange second) {
        final NodeType type = first.getNodeType();
        return matchLabel(first.getLabel(), second.getLabel(),
                properties.getLabelImportance(type), 4, 0.3, 0.1)
                * matchStates(first, second, CodeChange::getParent, properties::getAsParentImportance)
                * matchStates(first, second, MoveChange::getOldParent, properties::getAsParentImportance)
                * matchStates(first, second, CodeChange::getParentOfParent, properties::getAsParentOfParentImportance)
                * matchStates(first, second, MoveChange::getOldParentOfParent, properties::getAsParentOfParentImportance)
                * matchArrays(first.getChildren(), second.getChildren(),
                properties.getChildrenImportance(type), 0.1);
    }

    private <T extends CodeChange> double matchStates(T first, T second, Function<T, NodeState> extractor,
                                                      Function<NodeType, Double> importanceSupplier) {
        final NodeState firstState = extractor.apply(first);
        final NodeState secondState = extractor.apply(second);
        final NodeType firstType = firstState.getType();
        final NodeType secondType = secondState.getType();
        if (firstType != secondType) {
            return 1 - Math.max(importanceSupplier.apply(firstType), importanceSupplier.apply(secondType));
        }
        final double penalty = matchLabel(firstState.getLabel(), secondState.getLabel(),
                properties.getLabelImportance(firstType), 4, 0.3, 0.1);
        return withImportance(penalty, importanceSupplier.apply(firstType));
    }

    private static <T> double matchArrays(T[] first, T[] second, double importance, double penaltyForEdit) {
        final int errors = Arrays.equals(first, second) ? 0 :
                DamerauLevenshteinDistance.problemFor(first, second).solve();
        return withImportance(errors == 0 ? 1 : Math.pow(1 - penaltyForEdit, errors), importance);
    }


    private static <T> double matchLabel(String first, String second, double importance,
                                         int editingLimit, double editingPartLimit, double penaltyForEdit) {
        final int errors = first.equals(second) ? 0 :
                DamerauLevenshteinDistance.problemFor(first, second).solve();
        if (errors == 0) {
            return 1;
        }
        if (errors >= editingLimit || errors >= editingPartLimit * Math.min(first.length(), second.length())) {
            return 1 - importance;
        }
        return withImportance(Math.pow(1 - penaltyForEdit, errors), importance);
    }

    private static double withImportance(double value, double importance) {
        return 1 - importance * (1 - value);
    }

    @Override
    public double distance(CodeChange first, CodeChange second) {
        if (first.getChangeType() != second.getChangeType() || first.getNode().getType() != second.getNode().getType()) {
            return 1;
        }
        switch (first.getChangeType()) {
            case MOVE:
                return 1 - matchMoveChanges((MoveChange) first, (MoveChange) second);
            case DELETE:
                return 1 - matchDeleteChanges((DeleteChange) first, (DeleteChange) second);
            case UPDATE:
                return 1 - matchUpdateChanges((UpdateChange) first, (UpdateChange) second);
            case INSERT:
                return 1 - matchInsertChanges((InsertChange) first, (InsertChange) second);
            default:
                throw new RuntimeException("Unexpected enum type: " + first.getChangeType());
        }
    }

    public static int getChangeClass(CodeChange change) {
        return change.getChangeType().ordinal() + change.getNode().getType().ordinal() * 10;
    }

    private static ComparisionProperties defaultProperties() {
        final ComparisionProperties properties = new ComparisionProperties(
                0.8,
                0.6,
                0.7,
                0.4,
                0.4,
                0);

        properties.setLabelImportance(METHOD_INVOCATION, 1, 1);
        properties.setAsParentImportance(METHOD_INVOCATION, 1);
        properties.setAsParentOfParentImportance(METHOD_INVOCATION, 1);
        properties.setChildrenImportance(METHOD_INVOCATION, 0.9);

        properties.setLabelImportance(INFIX_EXPRESSION, 0.9, 0.9);
        properties.setAsParentImportance(INFIX_EXPRESSION, 0.9);
        properties.setAsParentOfParentImportance(INFIX_EXPRESSION, 0.9);
        properties.setChildrenImportance(INFIX_EXPRESSION, 0.9);

        properties.setLabelImportance(STRING_LITERAL, 0.9, 0.4);

        return properties;
    }
}
