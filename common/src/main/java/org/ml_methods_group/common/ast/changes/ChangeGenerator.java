package org.ml_methods_group.common.ast.changes;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.generation.ASTGenerator;

import java.io.Serializable;

public interface ChangeGenerator extends Serializable {
    Changes getChanges(Solution origin, Solution target);
    int diffSize(ITree origin, ITree target);
    ASTGenerator getGenerator();
}
