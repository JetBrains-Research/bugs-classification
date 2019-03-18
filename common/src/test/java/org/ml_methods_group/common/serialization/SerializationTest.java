package org.ml_methods_group.common.serialization;

import org.junit.Test;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.proto.*;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static java.util.Arrays.deepEquals;
import static org.junit.Assert.*;
import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.ast.NodeType.*;


public class SerializationTest {

    @FunctionalInterface
    private interface UnsafeBiConsumer<T, V> {
        void accept(T value1, V value2) throws Exception;
    }

    @FunctionalInterface
    private interface UnsafeFunction<T, V> {
        V apply(T value) throws Exception;
    }

    private static <T, V> T writeAndRead(T value,
                                         Function<T, V> mapper,
                                         Function<V, T> invertedMapper,
                                         UnsafeBiConsumer<V, OutputStream> writer,
                                         UnsafeFunction<InputStream, V> reader) throws Exception {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        writer.accept(mapper.apply(value), out);
        final ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
        return invertedMapper.apply(reader.apply(in));
    }

    private static final CodeChange CODE_CHANGE_EXAMPLE;

    static {
        final NodeState nodeBefore = new NodeState(MY_VARIABLE_NAME, "String", "String@1", "token", 0);
        final NodeState parentBefore = new NodeState(MY_METHOD_INVOCATION_ARGUMENTS, null, "parse", null, 1);
        final NodeState parentOfParentBefore = new NodeState(METHOD_INVOCATION, null, "parse", null, 0);
        final NodeState[] unclesBefore = {
                new NodeState(SIMPLE_NAME, null, "parse", null, 0),
                parentBefore
        };
        final NodeState[] brothersBefore = {
                nodeBefore,
                new NodeState(NUMBER_LITERAL, null, "10", null, 1),
                new NodeState(NULL_LITERAL, null, "", null, 2)
        };
        final NodeState[] childrenBefore = {};
        final NodeContext contextBefore = new NodeContext(nodeBefore, parentBefore, parentOfParentBefore,
                unclesBefore, brothersBefore, childrenBefore);

        final NodeState nodeAfter = new NodeState(MY_VARIABLE_NAME, "String", "String@2", "text", 0);
        final NodeState parentAfter = new NodeState(MY_METHOD_INVOCATION_ARGUMENTS, null, "parse", null, 1);
        final NodeState parentOfParentAfter = new NodeState(METHOD_INVOCATION, null, "parse", null, 0);
        final NodeState[] unclesAfter = {
                new NodeState(SIMPLE_NAME, null, "parse", null, 0),
                parentAfter
        };
        final NodeState[] brothersAfter = {
                nodeAfter,
                new NodeState(NUMBER_LITERAL, null, "100", null, 1),
                new NodeState(NULL_LITERAL, null, "", null, 2)
        };
        final NodeState[] childrenAfter = {};
        final NodeContext contextAfter = new NodeContext(nodeAfter, parentAfter, parentOfParentAfter,
                unclesAfter, brothersAfter, childrenAfter);
        CODE_CHANGE_EXAMPLE = new CodeChange(contextBefore, contextAfter, ChangeType.UPDATE);
    }

    @Test
    public void testSolutionTransformation() throws Exception {
        final Solution solution = new Solution("some code", 1, 2, 3, OK);
        final Solution parsedSolution = writeAndRead(solution,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transform,
                ProtoSolution::writeTo,
                ProtoSolution::parseFrom);
        assertEquals(solution, parsedSolution);
    }

    @Test
    public void testNodeStateTransformation() throws Exception {
        final NodeState state = new NodeState(MY_VARIABLE_NAME, "int", "int@1",
                "cnt", 0);
        final NodeState parsedState = writeAndRead(state,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transform,
                ProtoNodeState::writeTo,
                ProtoNodeState::parseFrom);
        assertEquals(state, parsedState);
    }

    @Test
    public void testNodeContextTransformation() throws Exception {
        final NodeContext context = CODE_CHANGE_EXAMPLE.getOriginalContext();
        final NodeContext parsedContext = writeAndRead(context,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transform,
                ProtoNodeContext::writeTo,
                ProtoNodeContext::parseFrom);
        assertEquals(context, parsedContext);
    }

    @Test
    public void testCodeChangeTransformation() throws Exception {
        final CodeChange codeChange = CODE_CHANGE_EXAMPLE;
        final CodeChange parsedChange = writeAndRead(codeChange,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transform,
                ProtoAtomicChange::writeTo,
                ProtoAtomicChange::parseFrom);
        assertEquals(codeChange, parsedChange);
    }

    @Test
    public void testChangesTransformation() throws Exception {
        final Changes changes = new Changes(
                new Solution("code before", 1, 1, 1, FAIL),
                new Solution("code after", 1, 1, 2, OK),
                Collections.singletonList(CODE_CHANGE_EXAMPLE));
        final Changes parsedChanges = writeAndRead(changes,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transform,
                ProtoChanges::writeTo,
                ProtoChanges::parseFrom);
        assertEquals(changes, parsedChanges);
        assertArrayEquals(changes.getChanges().toArray(), parsedChanges.getChanges().toArray());
    }

    @Test
    public void testVectorFeaturesTransformation() throws Exception {
        final double[] values = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1, 0.44, 1E-100};
        final double[] parsedValues = SerializationTest.writeAndRead(values,
                EntityToProtoUtils::transformVectorFeatures,
                ProtoToEntityUtils::transform,
                ProtoVectorFeatures::writeTo,
                ProtoVectorFeatures::parseFrom);
        assertArrayEquals(values, parsedValues, 0);
    }

    @Test
    public void testVectorsListFeaturesTransformation() throws Exception {
        final List<double[]> values = Arrays.asList(
                new double[]{Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1, 0.44, 1E-100},
                new double[]{-100, 0, 1E300, -0.5E30, 15, 42, 78},
                new double[]{});
        final List<double[]> parsedValues = SerializationTest.writeAndRead(values,
                EntityToProtoUtils::transformVectorListFeatures,
                ProtoToEntityUtils::transform,
                ProtoVectorListFeatures::writeTo,
                ProtoVectorListFeatures::parseFrom);
        assertTrue(deepEquals(values.toArray(), parsedValues.toArray()));
    }

    @Test
    public void testStringListFeaturesTransformation() throws Exception {
        final List<String> values = Arrays.asList("feature1", "feature2", "feature3");
        final List<String> parsedValues = SerializationTest.writeAndRead(values,
                EntityToProtoUtils::transformStringListFeatures,
                ProtoToEntityUtils::transform,
                ProtoStringListFeatures::writeTo,
                ProtoStringListFeatures::parseFrom);
        assertArrayEquals(values.toArray(), parsedValues.toArray());
    }

    @Test
    public void testChangeListFeaturesTransformation() throws Exception {
        final List<CodeChange> changes = Collections.singletonList(CODE_CHANGE_EXAMPLE);
        final List<CodeChange> parsedChanges = writeAndRead(changes,
                EntityToProtoUtils::transformChangeListFeatures,
                ProtoToEntityUtils::transform,
                ProtoChangeListFeatures::writeTo,
                ProtoChangeListFeatures::parseFrom);
        assertArrayEquals(changes.toArray(), parsedChanges.toArray());
    }

    @Test
    public void testVectorFeaturesWrapperTransformation() throws Exception {
        final var wrapper = new Wrapper<>(new double[]{1, 2, 3, 4},
                new Solution("text", 1, 1, 1, OK));
        final var parsedWrapper = writeAndRead(wrapper,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::extractVectorFeatures,
                ProtoFeaturesWrapper::writeTo,
                ProtoFeaturesWrapper::parseFrom);

        assertEquals(wrapper.getMeta(), parsedWrapper.getMeta());
        assertArrayEquals(wrapper.getFeatures(), parsedWrapper.getFeatures(), 0);
    }

    @Test
    public void testVectorListFeaturesWrapperTransformation() throws Exception {
        final var wrapper = new Wrapper<>(
                Arrays.asList(
                        new double[]{1, 2, 3, 4},
                        new double[]{-1, -2, -3, -4},
                        new double[]{0.1, 0.2, 0.3, 0.4}),
                new Solution("text", 1, 1, 1, OK));
        final var parsedWrapper = writeAndRead(wrapper,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::extractVectorListFeatures,
                ProtoFeaturesWrapper::writeTo,
                ProtoFeaturesWrapper::parseFrom);

        assertEquals(wrapper.getMeta(), parsedWrapper.getMeta());
        assertTrue(deepEquals(wrapper.getFeatures().toArray(), parsedWrapper.getFeatures().toArray()));
    }

    @Test
    public void testStringListFeaturesWrapperTransformation() throws Exception {
        final var wrapper = new Wrapper<>(
                Arrays.asList("feature1", "feature2", "feature3", "#$_()^6\\/?"),
                new Solution("text", 1, 1, 1, OK));
        final var parsedWrapper = writeAndRead(wrapper,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::extractStringListFeatures,
                ProtoFeaturesWrapper::writeTo,
                ProtoFeaturesWrapper::parseFrom);

        assertEquals(wrapper.getMeta(), parsedWrapper.getMeta());
        assertArrayEquals(wrapper.getFeatures().toArray(), parsedWrapper.getFeatures().toArray());
    }

    @Test
    public void testChangeListFeaturesWrapperTransformation() throws Exception {
        final var wrapper = new Wrapper<>(
                Arrays.asList(CODE_CHANGE_EXAMPLE, CODE_CHANGE_EXAMPLE),
                new Solution("text", 1, 1, 1, OK));
        final var parsedWrapper = writeAndRead(wrapper,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::extractChangeListFeatures,
                ProtoFeaturesWrapper::writeTo,
                ProtoFeaturesWrapper::parseFrom);

        assertEquals(wrapper.getMeta(), parsedWrapper.getMeta());
        assertArrayEquals(wrapper.getFeatures().toArray(), parsedWrapper.getFeatures().toArray());
    }

    @Test
    public void testSolutionClusterTransformation() throws Exception {
        final var cluster = new Cluster<>(Arrays.asList(
                new Solution("some code", 1, 1, 1, OK),
                new Solution("some other code", 1, 2, 4, OK),
                new Solution("some another code", 1, 3, 6, OK)
        ));
        final var parsedCluster = SerializationTest.writeAndRead(cluster,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformCluster(proto, ProtoToEntityUtils::extractSolutions),
                ProtoCluster::writeTo,
                ProtoCluster::parseFrom);
        assertArrayEquals(cluster.elementsCopy().toArray(), parsedCluster.elementsCopy().toArray());
    }

    @Test
    public void testVectorFeaturesWrapperClusterTransformation() throws Exception {
        final var cluster = new Cluster<>(Arrays.asList(
                new Wrapper<>(new double[]{1, 2, 3},
                        new Solution("some code1", 1, 1, 1, OK)),
                new Wrapper<>(new double[]{4, 5, 6},
                        new Solution("some code2", 1, 2, 4, OK)),
                new Wrapper<>(new double[]{7, 8, 9},
                        new Solution("some code3", 1, 3, 6, OK))));
        final var parsedCluster = SerializationTest.writeAndRead(cluster,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformCluster(proto, ProtoToEntityUtils::extractVectorFeatures),
                ProtoCluster::writeTo,
                ProtoCluster::parseFrom);
        assertEquals(cluster.size(), parsedCluster.size());
        final var original = cluster.iterator();
        final var parsed = parsedCluster.iterator();
        while (original.hasNext()) {
            final var originalElement = original.next();
            final var parsedElement = parsed.next();
            assertEquals(originalElement.getMeta(), parsedElement.getMeta());
            assertArrayEquals(originalElement.getFeatures(), parsedElement.getFeatures(), 0);
        }
    }

    @Test
    public void testVectorListFeaturesWrapperClusterTransformation() throws Exception {
        final var cluster = new Cluster<>(Arrays.asList(
                new Wrapper<>(Collections.singletonList(new double[]{1, 2, 3}),
                        new Solution("some code1", 1, 1, 1, OK)),
                new Wrapper<>(Collections.singletonList(new double[]{4, 5, 6}),
                        new Solution("some code2", 1, 2, 4, OK)),
                new Wrapper<>(Collections.singletonList(new double[]{7, 8, 9}),
                        new Solution("some code3", 1, 3, 6, OK))));
        final var parsedCluster = SerializationTest.writeAndRead(cluster,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformCluster(proto, ProtoToEntityUtils::extractVectorListFeatures),
                ProtoCluster::writeTo,
                ProtoCluster::parseFrom);
        assertEquals(cluster.size(), parsedCluster.size());
        final var original = cluster.iterator();
        final var parsed = parsedCluster.iterator();
        while (original.hasNext()) {
            final var originalElement = original.next();
            final var parsedElement = parsed.next();
            assertEquals(originalElement.getMeta(), parsedElement.getMeta());
            assertTrue(deepEquals(originalElement.getFeatures().toArray(), parsedElement.getFeatures().toArray()));
        }
    }

    @Test
    public void testStringListFeaturesWrapperClusterTransformation() throws Exception {
        final var cluster = new Cluster<>(Arrays.asList(
                new Wrapper<>(Arrays.asList("1", "2", "3"),
                        new Solution("some code1", 1, 1, 1, OK)),
                new Wrapper<>(Arrays.asList("4", "5", "6"),
                        new Solution("some code2", 1, 2, 4, OK)),
                new Wrapper<>(Arrays.asList("7", "8", "9"),
                        new Solution("some code3", 1, 3, 6, OK))));
        final var parsedCluster = SerializationTest.writeAndRead(cluster,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformCluster(proto, ProtoToEntityUtils::extractStringListFeatures),
                ProtoCluster::writeTo,
                ProtoCluster::parseFrom);
        assertEquals(cluster.size(), parsedCluster.size());
        final var original = cluster.iterator();
        final var parsed = parsedCluster.iterator();
        while (original.hasNext()) {
            final var originalElement = original.next();
            final var parsedElement = parsed.next();
            assertEquals(originalElement.getMeta(), parsedElement.getMeta());
            assertArrayEquals(originalElement.getFeatures().toArray(), parsedElement.getFeatures().toArray());
        }
    }

    @Test
    public void testChangeListFeaturesWrapperClusterTransformation() throws Exception {
        final var cluster = new Cluster<>(Arrays.asList(
                new Wrapper<>(Collections.singletonList(CODE_CHANGE_EXAMPLE),
                        new Solution("some code1", 1, 1, 1, OK)),
                new Wrapper<>(Collections.singletonList(CODE_CHANGE_EXAMPLE),
                        new Solution("some code2", 1, 2, 4, OK)),
                new Wrapper<>(Collections.singletonList(CODE_CHANGE_EXAMPLE),
                        new Solution("some code3", 1, 3, 6, OK))));
        final var parsedCluster = SerializationTest.writeAndRead(cluster,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformCluster(proto, ProtoToEntityUtils::extractChangeListFeatures),
                ProtoCluster::writeTo,
                ProtoCluster::parseFrom);
        assertEquals(cluster.size(), parsedCluster.size());
        final var original = cluster.iterator();
        final var parsed = parsedCluster.iterator();
        while (original.hasNext()) {
            final var originalElement = original.next();
            final var parsedElement = parsed.next();
            assertEquals(originalElement.getMeta(), parsedElement.getMeta());
            assertArrayEquals(originalElement.getFeatures().toArray(), parsedElement.getFeatures().toArray());
        }
    }

    @Test
    public void testSolutionClustersTransformation() throws Exception {
        final var clusters = new Clusters<>(Arrays.asList(
                new Cluster<>(Arrays.asList(
                        new Solution("some code1", 1, 1, 1, OK),
                        new Solution("some code2", 1, 2, 4, OK),
                        new Solution("some code3", 1, 3, 6, OK))),
                new Cluster<>(Arrays.asList(
                        new Solution("some code4", 2, 4, 8, OK),
                        new Solution("some code5", 2, 5, 10, OK),
                        new Solution("some code6", 2, 6, 12, OK)))));
        final var parsedClusters = SerializationTest.writeAndRead(clusters,
                EntityToProtoUtils::transform,
                proto -> ProtoToEntityUtils.transformClusters(proto, ProtoToEntityUtils::extractSolutions),
                ProtoClusters::writeTo,
                ProtoClusters::parseFrom);
        final var clustersElements = clusters.getClusters();
        final var parsedClustersElements = parsedClusters.getClusters();
        assertEquals(clustersElements.size(), parsedClustersElements.size());
        final var original = parsedClustersElements.iterator();
        final var parsed = clustersElements.iterator();
        while (original.hasNext()) {
            final var originalElement = original.next();
            final var parsedElement = parsed.next();
            assertArrayEquals(originalElement.elementsCopy().toArray(), parsedElement.elementsCopy().toArray());
        }
    }

    @Test
    public void testMarkedClustersTransformation() throws Exception {
        final List<Solution> solutions = Arrays.asList(
                new Solution("some code1", 1, 1, 1, OK),
                new Solution("some code2", 1, 2, 4, OK),
                new Solution("some code3", 1, 3, 6, OK),
                new Solution("some code4", 2, 4, 8, OK),
                new Solution("some code5", 2, 5, 10, OK),
                new Solution("some code6", 2, 6, 12, OK));
        final var clusters = new MarkedClusters<>(Map.of(
                new Cluster<>(solutions.subList(0, 3)),
                "mark1",
                new Cluster<>(solutions.subList(3, 6)),
                "mark2"));
        final var parsedClusters = SerializationTest.writeAndRead(clusters,
                EntityToProtoUtils::transform,
                ProtoToEntityUtils::transformMarkedClusters,
                ProtoMarkedClusters::writeTo,
                ProtoMarkedClusters::parseFrom);
        final Map<Solution, String> marks = parsedClusters.getFlatMarks();
        assertEquals("mark1", marks.get(solutions.get(0)));
        assertEquals("mark1", marks.get(solutions.get(1)));
        assertEquals("mark1", marks.get(solutions.get(2)));
        assertEquals("mark2", marks.get(solutions.get(3)));
        assertEquals("mark2", marks.get(solutions.get(4)));
        assertEquals("mark2", marks.get(solutions.get(5)));
    }
}
