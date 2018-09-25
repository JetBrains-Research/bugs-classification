package org.ml_methods_group.common.changes.generation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;

public interface ChangeGenerator {
    Changes getChanges(Solution origin, Solution target);
    ITree getTree(Solution solution);
}
