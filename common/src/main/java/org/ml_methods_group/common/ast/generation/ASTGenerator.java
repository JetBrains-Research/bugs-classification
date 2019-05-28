package org.ml_methods_group.common.ast.generation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;

import java.io.Serializable;

public interface ASTGenerator extends Serializable {
    ITree buildTree(Solution solution);
}
