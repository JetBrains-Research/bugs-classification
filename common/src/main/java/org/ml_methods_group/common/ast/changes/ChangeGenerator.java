package org.ml_methods_group.common.ast.changes;

import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.generation.ASTGenerator;

public interface ChangeGenerator {
    Changes getChanges(Solution origin, Solution target);
    ASTGenerator getGenerator();
}
