package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.TreeGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.changes.ChangeFilter.Resolution;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.NodeType;
import org.ml_methods_group.core.entities.Solution;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.changes.ChangeFilter.Resolution.ACCEPT_WITHOUT_LABEL;
import static org.ml_methods_group.core.changes.ChangeFilter.Resolution.REJECT;

public class SimpleChangeGenerator implements ChangeGenerator {
    private final Matchers matchers;
    private final TreeGenerator generator;
    private final ChangeFilter filter;
    private final Map<Integer, WeakReference<ITree>> cache = new HashMap<>();

    public SimpleChangeGenerator(ChangeFilter filter) {
        this.filter = filter;
        matchers = Matchers.getInstance();
        generator = new JdtTreeGenerator();
    }

    @Override
    public List<CodeChange> getChanges(Solution before, Solution after) {
        try {
            final ITree treeBefore = getTree(before);
            final ITree treeAfter = getTree(after);
            final Matcher matcher = matchers.getMatcher(treeBefore, treeAfter);
            matcher.match();
            final ActionGenerator actions = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
            return actions.generate().stream()
                    .map(action -> fromAction(action, before.getSolutionId(), after.getSolutionId()))
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private ITree getTree(Solution solution) throws IOException {
        final WeakReference<ITree> tree = cache.get(solution.getSolutionId());
        if (tree != null && tree.get() != null) {
            return tree.get();
        }
        final String code;
        if (!solution.getCode().contains("class ")) {
            code = "public class MY_MAGIC_CLASS_NAME {\n" + solution.getCode() + "\n}";
        } else {
            code = solution.getCode();
        }
        final ITree root = generator.generateFromString(code).getRoot();
        cache.put(solution.getSolutionId(), new WeakReference<ITree>(root));
        return root;
    }

    private static String normalize(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
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
        final Resolution resolution = filter.accept(insert);
        return resolution == REJECT ? null :
                CodeChange.createInsertChange(
                        originalId,
                        targetId,
                        getNodeType(node, 0),
                        getNodeType(node, 1),
                        getNodeType(node, 2),
                        resolution == ACCEPT_WITHOUT_LABEL ? "" : normalize(node.getLabel())
                );
    }

    private CodeChange fromAction(Delete delete, int originalId, int targetId) {
        final ITree node = delete.getNode();
        final Resolution resolution = filter.accept(delete);
        return resolution == REJECT ? null : CodeChange.createDeleteChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                resolution == ACCEPT_WITHOUT_LABEL ? "" : normalize(node.getLabel())
        );
    }


    private CodeChange fromAction(Move move, int originalId, int targetId) {
        final ITree node = move.getNode();
        final ITree parent = move.getParent();
        final Resolution resolution = filter.accept(move);
        return resolution == REJECT ? null : CodeChange.createMoveChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(parent, 0),
                getNodeType(parent, 1),
                getNodeType(node, 1),
                getNodeType(node, 2),
                resolution == ACCEPT_WITHOUT_LABEL ? "" : normalize(node.getLabel())
        );
    }

    private CodeChange fromAction(Update update, int originalId, int targetId) {
        final ITree node = update.getNode();
        final Resolution resolution = filter.accept(update);
        return resolution == REJECT ? null : CodeChange.createUpdateChange(
                originalId,
                targetId,
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                resolution == ACCEPT_WITHOUT_LABEL ? "" : normalize(update.getValue()),
                resolution == ACCEPT_WITHOUT_LABEL ? "" : normalize(node.getLabel())
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
