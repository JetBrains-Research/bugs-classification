package org.ml_methods_group.core.preparation;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.TreeGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.NodeType;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.entities.CodeChange;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class ChangesBuilder {
    private final Matchers matchers;
    private final TreeGenerator generator;

    public ChangesBuilder() {
        matchers = Matchers.getInstance();
        generator = new JdtTreeGenerator();
    }

    public List<CodeChange> findChanges(Solution before, Solution after) throws IOException {
        final String codeBefore;
        final String codeAfter;
        if (!before.getCode().contains("class ") && !after.getCode().contains("class ")) {
            codeBefore = "public class MY_MAGIC_CLASS_NAME {\n" + before.getCode() + "\n}";
            codeAfter = "public class MY_MAGIC_CLASS_NAME {\n" + after.getCode() + "\n}";
        } else {
            codeBefore = before.getCode();
            codeAfter = after.getCode();
        }
        final ITree treeBefore = generator.generateFromString(codeBefore).getRoot();
        final ITree treeAfter = generator.generateFromString(codeAfter).getRoot();
        final Matcher matcher = matchers.getMatcher(treeBefore, treeAfter);
        matcher.match();
        final ActionGenerator actions = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
        return actions.generate().stream()
                .map(action -> fromAction(action, before.getSolutionId(), after.getSolutionId()))
                .collect(Collectors.toList());
    }

    private static String normalize(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
    }

    private static CodeChange fromAction(Action action, int originalId, int targetId) {
        if (action.getClass() == Insert.class) {
            return fromAction((Insert) action, originalId, targetId);
        } else if (action.getClass() == Delete.class) {
            return fromAction((Delete) action, originalId, targetId);
        } else if (action.getClass() == Move.class) {
            return fromAction((Move) action, originalId, targetId);
        } else {
            return fromAction((Update) action, originalId, targetId);
        }
    }

    private static CodeChange fromAction(Insert insert, int originalId, int targetId) {
        final ITree node = insert.getNode();
        return CodeChange.createInsertChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel()),
                getPosition(node),
                node.getId()
        );
    }

    private static CodeChange fromAction(Delete delete, int originalId, int targetId) {
        final ITree node = delete.getNode();
        return CodeChange.createDeleteChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel()),
                getPosition(node),
                node.getId()
        );
    }


    private static CodeChange fromAction(Move move, int originalId, int targetId) {
        final ITree node = move.getNode();
        final ITree parent = move.getParent();
        return CodeChange.createMoveChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(parent, 0),
                getNodeType(parent, 1),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel()),
                getPosition(node),
                node.getId()
        );
    }

    private static CodeChange fromAction(Update update, int originalId, int targetId) {
        final ITree node = update.getNode();
        return CodeChange.createUpdateChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(update.getValue()),
                normalize(node.getLabel()),
                getPosition(node),
                node.getId()
        );
    }

    private static NodeType getNodeType(ITree node, int steps) {
        if (node == null) {
            return NodeType.NONE;
        }
        if (steps == 0) {
            return node.getType() < 0 ? NodeType.NONE : NodeType.valueOf(node.getType());
        }
        return getNodeType(node.getParent(), steps - 1);
    }

    private static int getPosition(ITree node) {
        final ITree parent = node.getParent();
        return parent == null ? 0 : Math.min(1 + parent.getChildPosition(node), 7);
    }
}
