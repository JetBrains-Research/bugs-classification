package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.Wrapper;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.proto.*;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class EntityToProtoUtils {
    public static ProtoSolution transform(Solution solution) {
        return ProtoSolution.newBuilder()
                .setCode(solution.getCode())
                .setProblemId(solution.getProblemId())
                .setSessionId(solution.getSessionId())
                .setSolutionId(solution.getSolutionId())
                .setVerdict(solution.getVerdict().ordinal())
                .build();
    }

    public static ProtoNodeState transform(NodeState state) {
        return ProtoNodeState.newBuilder()
                .setNodeType(state.getType().ordinal())
                .setJavaType(state.getJavaType())
                .setLabel(state.getLabel())
                .setOriginalLabel(state.getOriginalLabel())
                .setPositionInParent(state.getPositionInParent())
                .build();
    }

    public static List<ProtoNodeState> transform(NodeState[] state) {
        return Arrays.stream(state)
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
    }

    public static ProtoNodeContext transform(CodeChange.NodeContext context) {
        return ProtoNodeContext.newBuilder()
                .setNode(transform(context.getNode()))
                .setParent(transform(context.getParent()))
                .setParentOfParent(transform(context.getParentOfParent()))
                .addAllUncles(transform(context.getUncles()))
                .addAllBrothers(transform(context.getBrothers()))
                .addAllChildren(transform(context.getChildren()))
                .build();

    }

    public static ProtoAtomicChange transform(CodeChange change) {
        return ProtoAtomicChange.newBuilder()
                .setChangeType(change.getChangeType().ordinal())
                .setOriginalContext(transform(change.getOriginalContext()))
                .setDestinationContext(transform(change.getDestinationContext()))
                .build();
    }

    public static ProtoChanges transform(Changes changes) {
        final List<ProtoAtomicChange> atomicChanges = changes.getChanges().stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoChanges.newBuilder()
                .setSrc(transform(changes.getOrigin()))
                .setDst(transform(changes.getTarget()))
                .addAllChanges(atomicChanges)
                .build();
    }

    public static ProtoVectorFeatures transformVectorFeatures(double[] vector) {
        final ProtoVectorFeatures.Builder builder = ProtoVectorFeatures.newBuilder();
        for (var value : vector) {
            builder.addFeatures(value);
        }
        return builder.build();
    }

    public static ProtoVectorListFeatures transformVectorListFeatures(List<double[]> features) {
        final List<ProtoVectorFeatures> protoFeatures = features.stream()
                .map(EntityToProtoUtils::transformVectorFeatures)
                .collect(Collectors.toList());
        return ProtoVectorListFeatures.newBuilder()
                .addAllFeatures(protoFeatures)
                .build();
    }

    public static ProtoStringListFeatures transformStringListFeatures(List<String> features) {
        return ProtoStringListFeatures.newBuilder()
                .addAllFeatures(features)
                .build();
    }

    public static ProtoChangeListFeatures transformChangeListFeatures(List<CodeChange> features) {
        final List<ProtoAtomicChange> protoFeatures = features.stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoChangeListFeatures.newBuilder()
                .addAllFeatures(protoFeatures)
                .build();
    }

    private static <F> void tryCastFeaturesList(List<?> features, Class<F> template,
                                               Consumer<List<F>> consumer) {
        final boolean classesCheck = features.stream()
                .map(Object::getClass)
                .allMatch(template::equals);
        if (classesCheck) {
            final List<F> result = features.stream().map(template::cast)
                    .collect(Collectors.toList());
            consumer.accept(result);
        }
    }

    public static ProtoFeaturesWrapper transform(Wrapper<?, ?> wrapper) {
        final ProtoFeaturesWrapper.Builder builder = ProtoFeaturesWrapper.newBuilder();
        if (wrapper.getMeta() instanceof Solution) {
            builder.setSolution(transform((Solution) wrapper.getMeta()));
        } else {
            throw new IllegalArgumentException("Unsupported meta information type!");
        }
        if (wrapper.getFeatures() instanceof double[]) {
            builder.setVectorFeatures(transformVectorFeatures((double[]) wrapper.getFeatures()));
        } else if (wrapper.getFeatures() instanceof List) {
            final List<?> features = (List<?>) wrapper.getFeatures();
            tryCastFeaturesList(features, String.class, x ->
                    builder.setStringListFeatures(transformStringListFeatures(x)));
            tryCastFeaturesList(features, CodeChange.class, x ->
                    builder.setChangeListFeatures(transformChangeListFeatures(x)));
            tryCastFeaturesList(features, double[].class, x ->
                    builder.setVectorListFeatures(transformVectorListFeatures(x)));
        }
        return builder.build();
    }

    public static ProtoFeaturesWrapper wrapAndTransform(Solution solution) {
        return ProtoFeaturesWrapper.newBuilder()
                .setSolution(transform(solution))
                .build();
    }

    public static ProtoCluster transform(Cluster<?> cluster) {
        final List<ProtoFeaturesWrapper> solutions;
        if (cluster.stream().map(Object::getClass).allMatch(Solution.class::equals)) {
            solutions = cluster.stream()
                    .map(Solution.class::cast)
                    .map(EntityToProtoUtils::wrapAndTransform)
                    .collect(Collectors.toList());
        } else if (cluster.stream().map(Object::getClass).allMatch(Wrapper.class::equals)) {
            solutions = cluster.stream()
                    .map(Wrapper.class::cast)
                    .map(EntityToProtoUtils::transform)
                    .collect(Collectors.toList());
        } else {
            throw new IllegalArgumentException("Unsupported cluster elements types");
        }
        return ProtoCluster.newBuilder()
                .addAllSolutions(solutions)
                .build();
    }

    public static ProtoClusters transform(Clusters<?> clusters) {
        final List<ProtoCluster> proto = clusters.getClusters()
                .stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoClusters.newBuilder()
                .addAllClusters(proto)
                .build();
    }
}
