package org.ml_methods_group.common.ast.generation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;

public interface ASTGenerator {
    ITree buildTree(Solution solution);
}
