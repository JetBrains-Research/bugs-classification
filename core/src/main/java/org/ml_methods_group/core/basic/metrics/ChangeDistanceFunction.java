package org.ml_methods_group.core.basic.metrics;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.changes.CodeChange;

import java.util.function.Function;

public class ChangeDistanceFunction implements DistanceFunction<CodeChange> {

    private double matchInsertChanges(CodeChange first, CodeChange second) {
        if (first.getLabel().isEmpty() && second.getLabel().isEmpty()) {
            return check(first, second, CodeChange::getParentType, 0.6)
                    + check(first, second, CodeChange::getParentOfParentType, 0.4);
        }
        return check(first, second, CodeChange::getLabel, 0.5)
                + check(first, second, CodeChange::getParentType, 0.3)
                + check(first, second, CodeChange::getParentOfParentType, 0.2);
    }

    private double matchDeleteChanges(CodeChange first, CodeChange second) {
        if (first.getLabel().isEmpty() && second.getLabel().isEmpty()) {
            return check(first, second, CodeChange::getParentType, 0.6)
                    + check(first, second, CodeChange::getParentOfParentType, 0.4);
        }
        return check(first, second, CodeChange::getLabel, 0.5)
                + check(first, second, CodeChange::getParentType, 0.3)
                + check(first, second, CodeChange::getParentOfParentType, 0.2);
    }

    private double matchUpdateChanges(CodeChange first, CodeChange second) {
        return check(first, second, CodeChange::getLabel, 0.4)
                + check(first, second, CodeChange::getOldLabel, 0.2)
                + check(first, second, CodeChange::getParentType, 0.25)
                + check(first, second, CodeChange::getParentOfParentType, 0.15);
    }

    private double matchMoveChanges(CodeChange first, CodeChange second) {
        return check(first, second, CodeChange::getLabel, 0.3)
                + check(first, second, CodeChange::getParentType, 0.25)
                + check(first, second, CodeChange::getParentOfParentType, 0.1)
                + check(first, second, CodeChange::getOldParentType, 0.25)
                + check(first, second, CodeChange::getOldParentOfParentType, 0.1);
    }

    private static <T> double check(T first, T second, Function<T, ?> extractor, double penalty) {
        return extractor.apply(first).equals(extractor.apply(second)) ? 0 : penalty;
    }

    @Override
    public double distance(CodeChange first, CodeChange second) {
        if (first.getChangeType() != second.getChangeType() || first.getNodeType() != second.getNodeType()) {
            return 1;
        }
        switch (first.getChangeType()) {
            case MOVE:
                return matchMoveChanges(first, second);
            case DELETE:
                return matchDeleteChanges(first, second);
            case UPDATE:
                return matchUpdateChanges(first, second);
            case INSERT:
                return matchInsertChanges(first, second);
            default:
                throw new RuntimeException("Unexpected enum type: " + first.getChangeType());
        }
    }

    public static int getChangeClass(CodeChange change) {
        return change.getChangeType().ordinal() + change.getNodeType().ordinal() * 10;
    }
}
