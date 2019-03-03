package org.ml_methods_group.common.ast.normalization;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.MetadataKeys;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.ml_methods_group.common.ast.ASTUtils.getFirstChild;
import static org.ml_methods_group.common.ast.NodeType.*;


public class BasicASTNormalizer implements ASTNormalizer {

    public void normalize(TreeContext context, String code) {
        context.setRoot(new StructureProcessor(context, code).visit(context.getRoot()));
        context.validate();
    }

    protected static class StructureProcessor extends ASTProcessor {
        final ArrayDeque<Map<String, String>> variablesTypes = new ArrayDeque<>();
        final ArrayDeque<String> typeDeclarations = new ArrayDeque<>();
        final String code;

        StructureProcessor(TreeContext context, String code) {
            super(context);
            this.code = code;
            variablesTypes.add(new HashMap<>());
        }

        void register(String name) {
            assert !typeDeclarations.isEmpty();
            register(name, typeDeclarations.peekLast());
        }

        void registerAsArray(String name, int arity) {
            assert !typeDeclarations.isEmpty();
            final String type = IntStream.range(0, arity)
                    .mapToObj(x -> "[]")
                    .collect(Collectors.joining("", typeDeclarations.peekLast(), ""));
            register(name, type);
        }

        void register(String name, String type) {
            variablesTypes.peekLast().put(name, type);
        }

        private String getVariableType(String name) {
            final Iterator<Map<String, String>> iterator = variablesTypes.descendingIterator();
            while (iterator.hasNext()) {
                final String alias = iterator.next().get(name);
                if (alias != null) {
                    return alias;
                }
            }
            return null;
        }

        void pushLayer() {
            variablesTypes.addLast(new HashMap<>());
        }

         void popLayer() {
            variablesTypes.pollLast();
        }

        private void pushTypeDeclaration(String type) {
            typeDeclarations.addLast(type);
        }

        private void popTypeDeclaration() {
            typeDeclarations.pollFirst();
        }

        @Override
        protected ITree visitFieldDeclaration(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitLambdaExpression(ITree node) {
            pushLayer();
            pushTypeDeclaration("UnknownLambdaArgumentType");
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            popLayer();
            return result;
        }

        @Override
        protected ITree visitQualifiedName(ITree node) {
            assert node.getChildren().isEmpty();
            if (node.getParent().getType() == NodeType.PACKAGE_DECLARATION.ordinal()) {
                return createNode(NodeType.SIMPLE_NAME, node.getLabel());
            }
            final String label = node.getLabel();
            final ITree path = createNode(NodeType.SIMPLE_NAME, label.substring(0, label.lastIndexOf('.')));
            final ITree member = createNode(NodeType.MY_MEMBER_NAME, label.substring(label.lastIndexOf('.') + 1));
            member.setChildren(Collections.singletonList(path));
            return visit(member);
        }

        @Override
        protected ITree visitVariableDeclarationFragment(ITree node) {
            final List<ITree> children = node.getChildren();
            assert children.get(0).getType() == NodeType.SIMPLE_NAME.ordinal();
            final String name = children.get(0).getLabel();
            final int arity = (int) children.stream()
                    .filter(CommonUtils.check(ITree::getType, type -> type == NodeType.DIMENSION.ordinal()))
                    .count();
            if (arity != 0) {
                registerAsArray(name, arity);
            } else {
                register(name);
            }
            final List<ITree> generated = new ArrayList<>();
            generated.add(visit(createNode(MY_VARIABLE_NAME, name)));
            children.subList(1, children.size())
                    .forEach(tree -> generated.add(visit(tree)));
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitVariableDeclarationStatement(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            final String typeName = getTypeName(type);
            pushTypeDeclaration(typeName);
            node.setLabel(typeName);
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitVariableDeclarationExpression(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            final String typeName = getTypeName(type);
            pushTypeDeclaration(typeName);
            final ITree result = defaultVisit(node);
            result.setLabel(typeName);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitSimpleType(ITree node) {
            final String label = node.getLabel();
            node.setLabel(label.substring(label.lastIndexOf('.') + 1));
            node.setChildren(Collections.emptyList());
            return node;
        }

        @Override
        protected ITree visitSimpleName(ITree node) {
            final String type = getVariableType(node.getLabel());
            if (type == null) {
                return node;
            }
            final ITree result = createNode(NodeType.MY_VARIABLE_NAME, node.getLabel());
            result.setMetadata(MetadataKeys.JAVA_TYPE, type);
            return visit(result);
        }

        @Override
        protected ITree visitExpressionStatement(ITree node) {
            assert node.getChildren().size() == 1;
            return visit(node.getChild(0));
        }

        @Override
        protected ITree visitMethodInvocation(ITree node) {
            final List<ITree> children = node.getChildren();
            final String text = code.substring(node.getPos(), node.getEndPos());
            final int bound;
            if (children.get(0).getType() != NodeType.SIMPLE_NAME.ordinal()) {
                bound = 0;
            } else if (text.matches("(?s)\\s*[a-zA-Z0-9_]+\\s*\\(.*")) {
                bound = 0;
            } else {
                bound = 1;
            }
            final List<ITree> generated = new ArrayList<>(children.size());
            final List<ITree> arguments = new ArrayList<>(children.size());
            boolean flag = false;
            for (int i = 0; i < children.size(); i++) {
                final ITree child = children.get(i);
                if (!flag && i >= bound && child.getType() == NodeType.SIMPLE_NAME.ordinal()) {
                    flag = true;
                    node.setLabel(child.getLabel());
                    continue;
                }
                final ITree result = visit(child);
                if (result == null) {
                    continue;
                }
                if (flag) {
                    arguments.add(result);
                } else {
                    generated.add(result);
                }
            }
            final ITree argumentsNode = createNode(NodeType.MY_METHOD_INVOCATION_ARGUMENTS, node.getLabel());
            argumentsNode.setChildren(arguments);
            generated.add(argumentsNode);
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitImportDeclaration(ITree node) {
            final List<ITree> children = node.getChildren();
            assert children.size() == 1;
            final List<ITree> generated = new ArrayList<>(2);
            final String name = children.get(0).getLabel();
            final int separatorIndex = name.lastIndexOf('.');
            if (separatorIndex == -1) {
                generated.add(visit(createNode(SIMPLE_TYPE, name)));
            } else {
                final String path = name.substring(0, separatorIndex);
                final String type = name.substring(separatorIndex + 1);
                generated.add(visit(createNode(NodeType.MY_PATH_NAME, path)));
                if (type.equals("*")) {
                    generated.add(visit(createNode(NodeType.MY_ALL_CLASSES, "")));
                } else {
                    generated.add(visit(createNode(SIMPLE_TYPE, type)));
                }
            }
            generated.removeIf(Objects::isNull);
            node.setChildren(generated);
            return super.visitImportDeclaration(node);
        }

        @Override
        protected ITree visitJavadoc(ITree node) {
            return null;
        }

        @Override
        protected ITree visitLineComment(ITree node) {
            return null;
        }

        @Override
        protected ITree visitBlockComment(ITree node) {
            return null;
        }

        @Override
        protected ITree visitSingleVariableDeclaration(ITree node) {
            final ITree name = getFirstChild(node, NodeType.SIMPLE_NAME);
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE,
                    ARRAY_TYPE, NodeType.UNION_TYPE);
            assert name != null;
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            register(name.getLabel());
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitBlock(ITree node) {
            pushLayer();
            final ITree result = defaultVisit(node);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitForStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 3);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitWhileStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 1);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitIfStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 1, 2);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitEnhancedForStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 2);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitUnionType(ITree node) {
            final List<ITree> children = node.getChildren()
                    .stream()
                    .map(this::visit)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
            node.setChildren(children);
            final String label = Arrays.stream(node.getLabel().split("\\|"))
                    .sorted()
                    .collect(Collectors.joining("|"));
            node.setLabel(label);
            return node;
        }

        private static String getTypeName(ITree node) {
            final NodeType type = NodeType.valueOf(node.getType());
            if (type == null) {
                throw new RuntimeException();
            }
            switch (type) {
                case PRIMITIVE_TYPE:
                    return node.getLabel();
                case SIMPLE_TYPE:
                    final String label = node.getLabel();
                    return label.substring(label.lastIndexOf('.') + 1);
                case PARAMETERIZED_TYPE:
                    final ITree child = getFirstChild(node, SIMPLE_TYPE);
                    assert child != null;
                    return child.getLabel();
                case ARRAY_TYPE:
                    return node.getLabel();
                case UNION_TYPE:
                    return "UnionThrowable";
            }
            throw new RuntimeException("Unexpected type!");
        }

        private ITree checkBlocks(ITree node, int... positions) {
            final BitSet targets = new BitSet();
            Arrays.stream(positions).forEach(targets::set);
            final List<ITree> children = node.getChildren();
            final List<ITree> generated = new ArrayList<>();
            for (int i = 0; i < children.size(); i++) {
                final ITree child = visit(children.get(i));
                if (targets.get(i) && child.getType() != NodeType.BLOCK.ordinal()) {
                    final ITree wrapper = createNode(NodeType.BLOCK, "");
                    wrapper.addChild(child);
                    generated.add(wrapper);
                } else {
                    generated.add(child);
                }

            }
            node.setChildren(generated);
            return node;
        }
    }
}
