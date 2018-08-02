package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.NodeType;

import java.util.Arrays;
import java.util.BitSet;

import static org.ml_methods_group.core.entities.NodeType.*;

public class ASTUtils {
    static ITree getFirstChild(ITree parent, NodeType... types) {
        BitSet acceptable = new BitSet();
        Arrays.stream(types)
                .mapToInt(NodeType::ordinal)
                .forEach(acceptable::set);
        for (ITree child : parent.getChildren()) {
            if (acceptable.get(child.getType())) {
                return child;
            }
        }
        return null;
    }
}