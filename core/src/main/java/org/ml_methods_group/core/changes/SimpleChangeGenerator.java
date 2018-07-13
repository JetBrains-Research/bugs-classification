package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.TreeGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.NodeType;
import org.ml_methods_group.core.entities.Solution;

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class SimpleChangeGenerator implements ChangeGenerator {
    private final Matchers matchers;
    private final TreeGenerator generator;
    private final ChangeFilter filter;
    private final LabelNormalizer normalizer;

    public SimpleChangeGenerator(ChangeFilter filter, LabelNormalizer normalizer) {
        this.filter = filter;
        this.normalizer = normalizer;
        matchers = Matchers.getInstance();
        generator = new JdtTreeGenerator();
    }

    @Override
    public List<CodeChange> getChanges(Solution before, Solution after) {
        final ITree treeBefore = getTree(before);
        final ITree treeAfter = getTree(after);
        final Matcher matcher = matchers.getMatcher(treeBefore, treeAfter);
        matcher.match();
        final ActionGenerator actions = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
        return actions.generate().stream()
                .map(action -> fromAction(action, before.getSolutionId(), after.getSolutionId()))
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    @Override
    public ITree getTree(Solution solution) {
        try {
            final String code;
            if (!solution.getCode().contains("class ")) {
                code = "public class MY_MAGIC_CLASS_NAME {\n" + solution.getCode() + "\n}";
            } else {
                code = solution.getCode();
            }
            return generator.generateFromString(code).getRoot();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private CodeChange fromAction(Action action, int originalId, int targetId) {
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

    private CodeChange fromAction(Insert insert, int originalId, int targetId) {
        final ITree node = insert.getNode();
        return !filter.accept(insert) ? null :
                CodeChange.createInsertChange(
                        originalId,
                        targetId,
                        getNodeType(node, 0),
                        getNodeType(node, 1),
                        getNodeType(node, 2),
                        normalizer.normalize(node.getLabel(), insert)
                );
    }

    private CodeChange fromAction(Delete delete, int originalId, int targetId) {
        final ITree node = delete.getNode();
        return !filter.accept(delete) ? null : CodeChange.createDeleteChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalizer.normalize(node.getLabel(), delete)
        );
    }


    private CodeChange fromAction(Move move, int originalId, int targetId) {
        final ITree node = move.getNode();
        final ITree parent = move.getParent();
        return !filter.accept(move) ? null : CodeChange.createMoveChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(parent, 0),
                getNodeType(parent, 1),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalizer.normalize(node.getLabel(), move)
        );
    }

    private CodeChange fromAction(Update update, int originalId, int targetId) {
        final ITree node = update.getNode();
        return !filter.accept(update) ? null : CodeChange.createUpdateChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalizer.normalize(update.getValue(), update),
                normalizer.normalize(node.getLabel(), update)
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
}
