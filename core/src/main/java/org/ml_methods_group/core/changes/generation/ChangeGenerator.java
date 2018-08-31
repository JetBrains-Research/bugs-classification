package org.ml_methods_group.core.changes.generation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.Solution;

public interface ChangeGenerator {
    Changes getChanges(Solution origin, Solution target);
    ITree getTree(Solution solution);
}
