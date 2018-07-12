package org.ml_methods_group.core.changes;

import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;

import java.util.List;

public interface ChangeGenerator {
    List<CodeChange> getChanges(Solution origin, Solution target);
}
