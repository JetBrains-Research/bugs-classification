package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.ITree;

import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

public class ASTUtils {
    public static ITree getFirstChild(ITree parent, NodeType... types) {
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

    public static boolean deepEquals(ITree first, ITree second) {
        if (first.getHash() != second.getHash()
                || first.getType() != second.getType()
                || !first.getLabel().equals(second.getLabel())) {
            return false;
        }
        final List<ITree> firstChildren = first.getChildren();
        final List<ITree> secondChildren = second.getChildren();
        if (firstChildren.size() != secondChildren.size()) {
            return false;
        }
        for (int i = 0; i < firstChildren.size(); i++) {
            if (!deepEquals(firstChildren.get(i), secondChildren.get(i))) {
                return false;
            }
        }
        return true;
    }
}
