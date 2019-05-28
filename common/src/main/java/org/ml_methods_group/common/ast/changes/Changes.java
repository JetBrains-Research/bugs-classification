package org.ml_methods_group.common.ast.changes;

import org.ml_methods_group.common.Solution;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

public class Changes implements Serializable {
    private final Solution origin;
    private final Solution target;
    private final List<CodeChange> changes;

    public Changes(Solution origin, Solution target, List<CodeChange> changes) {
        this.origin = origin;
        this.target = target;
        this.changes = changes;
    }

    public Solution getOrigin() {
        return origin;
    }

    public Solution getTarget() {
        return target;
    }

    public List<CodeChange> getChanges() {
        return changes;
    }

    @Override
    public int hashCode() {
        return Objects.hash(origin, target);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        final Changes changes = (Changes) o;

        return origin.equals(changes.origin) && target.equals(changes.target);
    }
}
