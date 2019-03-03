package org.ml_methods_group.common.ast.normalization;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.common.ast.NodeType;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public abstract class ASTProcessor {

    private final TreeContext context;

    protected ASTProcessor(TreeContext context) {
        this.context = context;
    }

    protected ITree createNode(NodeType type, String label) {
        return context.createTree(type.ordinal(), label, type.humanReadableName);
    }

    protected ITree visit(ITree node) {
        final NodeType type = NodeType.valueOf(node.getType());
        if (type == null) {
            throw new IllegalStateException();
        }
        switch (type) {
            case NONE:
                return visitNone(node);
            case ANONYMOUS_CLASS_DECLARATION:
                return visitAnonymousClassDeclaration(node);
            case ARRAY_ACCESS:
                return visitArrayAccess(node);
            case ARRAY_CREATION:
                return visitArrayCreation(node);
            case ARRAY_INITIALIZER:
                return visitArrayInitializer(node);
            case ARRAY_TYPE:
                return visitArrayType(node);
            case ASSERT_STATEMENT:
                return visitAssertStatement(node);
            case ASSIGNMENT:
                return visitAssignment(node);
            case BLOCK:
                return visitBlock(node);
            case BOOLEAN_LITERAL:
                return visitBooleanLiteral(node);
            case BREAK_STATEMENT:
                return visitBreakStatement(node);
            case CAST_EXPRESSION:
                return visitCastExpression(node);
            case CATCH_CLAUSE:
                return visitCatchClause(node);
            case CHARACTER_LITERAL:
                return visitCharacterLiteral(node);
            case CLASS_INSTANCE_CREATION:
                return visitClassInstanceCreation(node);
            case COMPILATION_UNIT:
                return visitCompilationUnit(node);
            case CONDITIONAL_EXPRESSION:
                return visitConditionalExpression(node);
            case CONSTRUCTOR_INVOCATION:
                return visitConstructorInvocation(node);
            case CONTINUE_STATEMENT:
                return visitContinueStatement(node);
            case DO_STATEMENT:
                return visitDoStatement(node);
            case EMPTY_STATEMENT:
                return visitEmptyStatement(node);
            case EXPRESSION_STATEMENT:
                return visitExpressionStatement(node);
            case FIELD_ACCESS:
                return visitFieldAccess(node);
            case FIELD_DECLARATION:
                return visitFieldDeclaration(node);
            case FOR_STATEMENT:
                return visitForStatement(node);
            case IF_STATEMENT:
                return visitIfStatement(node);
            case IMPORT_DECLARATION:
                return visitImportDeclaration(node);
            case INFIX_EXPRESSION:
                return visitInfixExpression(node);
            case INITIALIZER:
                return visitInitializer(node);
            case JAVADOC:
                return visitJavadoc(node);
            case LABELED_STATEMENT:
                return visitLabeledStatement(node);
            case METHOD_DECLARATION:
                return visitMethodDeclaration(node);
            case METHOD_INVOCATION:
                return visitMethodInvocation(node);
            case NULL_LITERAL:
                return visitNullLiteral(node);
            case NUMBER_LITERAL:
                return visitNumberLiteral(node);
            case PACKAGE_DECLARATION:
                return visitPackageDeclaration(node);
            case PARENTHESIZED_EXPRESSION:
                return visitParenthesizedExpression(node);
            case POSTFIX_EXPRESSION:
                return visitPostfixExpression(node);
            case PREFIX_EXPRESSION:
                return visitPrefixExpression(node);
            case PRIMITIVE_TYPE:
                return visitPrimitiveType(node);
            case QUALIFIED_NAME:
                return visitQualifiedName(node);
            case RETURN_STATEMENT:
                return visitReturnStatement(node);
            case SIMPLE_NAME:
                return visitSimpleName(node);
            case SIMPLE_TYPE:
                return visitSimpleType(node);
            case SINGLE_VARIABLE_DECLARATION:
                return visitSingleVariableDeclaration(node);
            case STRING_LITERAL:
                return visitStringLiteral(node);
            case SUPER_CONSTRUCTOR_INVOCATION:
                return visitSuperConstructorInvocation(node);
            case SUPER_FIELD_ACCESS:
                return visitSuperFieldAccess(node);
            case SUPER_METHOD_INVOCATION:
                return visitSuperMethodInvocation(node);
            case SWITCH_CASE:
                return visitSwitchCase(node);
            case SWITCH_STATEMENT:
                return visitSwitchStatement(node);
            case SYNCHRONIZED_STATEMENT:
                return visitSynchronizedStatement(node);
            case THIS_EXPRESSION:
                return visitThisExpression(node);
            case THROW_STATEMENT:
                return visitThrowStatement(node);
            case TRY_STATEMENT:
                return visitTryStatement(node);
            case TYPE_DECLARATION:
                return visitTypeDeclaration(node);
            case TYPE_DECLARATION_STATEMENT:
                return visitTypeDeclarationStatement(node);
            case TYPE_LITERAL:
                return visitTypeLiteral(node);
            case VARIABLE_DECLARATION_EXPRESSION:
                return visitVariableDeclarationExpression(node);
            case VARIABLE_DECLARATION_FRAGMENT:
                return visitVariableDeclarationFragment(node);
            case VARIABLE_DECLARATION_STATEMENT:
                return visitVariableDeclarationStatement(node);
            case WHILE_STATEMENT:
                return visitWhileStatement(node);
            case INSTANCEOF_EXPRESSION:
                return visitInstanceofExpression(node);
            case LINE_COMMENT:
                return visitLineComment(node);
            case BLOCK_COMMENT:
                return visitBlockComment(node);
            case TAG_ELEMENT:
                return visitTagElement(node);
            case TEXT_ELEMENT:
                return visitTextElement(node);
            case MEMBER_REF:
                return visitMemberRef(node);
            case METHOD_REF:
                return visitMethodRef(node);
            case METHOD_REF_PARAMETER:
                return visitMethodRefParameter(node);
            case ENHANCED_FOR_STATEMENT:
                return visitEnhancedForStatement(node);
            case ENUM_DECLARATION:
                return visitEnumDeclaration(node);
            case ENUM_CONSTANT_DECLARATION:
                return visitEnumConstantDeclaration(node);
            case TYPE_PARAMETER:
                return visitTypeParameter(node);
            case PARAMETERIZED_TYPE:
                return visitParameterizedType(node);
            case QUALIFIED_TYPE:
                return visitQualifiedType(node);
            case WILDCARD_TYPE:
                return visitWildcardType(node);
            case NORMAL_ANNOTATION:
                return visitNormalAnnotation(node);
            case MARKER_ANNOTATION:
                return visitMarkerAnnotation(node);
            case SINGLE_MEMBER_ANNOTATION:
                return visitSingleMemberAnnotation(node);
            case MEMBER_VALUE_PAIR:
                return visitMemberValuePair(node);
            case ANNOTATION_TYPE_DECLARATION:
                return visitAnnotationTypeDeclaration(node);
            case ANNOTATION_TYPE_MEMBER_DECLARATION:
                return visitAnnotationTypeMemberDeclaration(node);
            case MODIFIER:
                return visitModifier(node);
            case UNION_TYPE:
                return visitUnionType(node);
            case DIMENSION:
                return visitDimension(node);
            case LAMBDA_EXPRESSION:
                return visitLambdaExpression(node);
            case INTERSECTION_TYPE:
                return visitIntersectionType(node);
            case NAME_QUALIFIED_TYPE:
                return visitNameQualifiedType(node);
            case CREATION_REFERENCE:
                return visitCreationReference(node);
            case EXPRESSION_METHOD_REFERENCE:
                return visitExpressionMethodReference(node);
            case SUPER_METHOD_REFERENCE:
                return visitSuperMethodReference(node);
            case TYPE_METHOD_REFERENCE:
                return visitTypeMethodReference(node);

            case MY_MEMBER_NAME:
                return visitMyMemberName(node);
            case MY_ALL_CLASSES:
                return visitMyAllClasses(node);
            case MY_PATH_NAME:
                return visitMyPathName(node);
            case MY_METHOD_INVOCATION_ARGUMENTS:
                return visitMyMethodInvocationArguments(node);
            case MY_VARIABLE_NAME:
                return visitMyVariableName(node);
            default:
                throw new RuntimeException("Unexpected node type");
        }
    }

    protected ITree visitNone(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitAnonymousClassDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitArrayAccess(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitArrayCreation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitArrayInitializer(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitArrayType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitAssertStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitAssignment(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitBlock(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitBooleanLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitBreakStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitCastExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitCatchClause(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitCharacterLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitClassInstanceCreation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitCompilationUnit(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitConditionalExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitConstructorInvocation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitContinueStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitDoStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitEmptyStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitExpressionStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitFieldAccess(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitFieldDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitForStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitIfStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitImportDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitInfixExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitInitializer(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitJavadoc(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitLabeledStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMethodDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMethodInvocation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitNullLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitNumberLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitPackageDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitParenthesizedExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitPostfixExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitPrefixExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitPrimitiveType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitQualifiedName(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitReturnStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSimpleName(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSimpleType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSingleVariableDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitStringLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSuperConstructorInvocation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSuperFieldAccess(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSuperMethodInvocation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSwitchCase(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSwitchStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSynchronizedStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitThisExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitThrowStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTryStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTypeDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTypeDeclarationStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTypeLiteral(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitVariableDeclarationExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitVariableDeclarationFragment(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitVariableDeclarationStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitWhileStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitInstanceofExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitLineComment(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitBlockComment(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTagElement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTextElement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMemberRef(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMethodRef(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMethodRefParameter(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitEnhancedForStatement(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitEnumDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitEnumConstantDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTypeParameter(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitParameterizedType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitQualifiedType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitWildcardType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitNormalAnnotation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMarkerAnnotation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSingleMemberAnnotation(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMemberValuePair(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitAnnotationTypeDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitAnnotationTypeMemberDeclaration(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitModifier(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitUnionType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitDimension(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitLambdaExpression(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitIntersectionType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitNameQualifiedType(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitCreationReference(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitExpressionMethodReference(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitSuperMethodReference(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitTypeMethodReference(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMyMemberName(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMyPathName(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMyAllClasses(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMyMethodInvocationArguments(ITree node) {
        return defaultVisit(node);
    }

    protected ITree visitMyVariableName(ITree node) {
        return defaultVisit(node);
    }

    protected ITree defaultVisit(ITree node) {
        final List<ITree> children = node.getChildren();
        final List<ITree> generated = children.stream()
                .map(this::visit)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        if (!children.equals(generated)) {
            node.setChildren(generated);
        }
        return node;
    }
}
