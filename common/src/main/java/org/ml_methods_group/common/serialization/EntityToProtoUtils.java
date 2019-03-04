package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Cluster;
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
    public static ProtoSolution mapSolutionToProto(Solution solution) {
        return ProtoSolution.newBuilder()
                .setCode(solution.getCode())
                .setProblemId(solution.getProblemId())
                .setSessionId(solution.getSessionId())
                .setSolutionId(solution.getSolutionId())
                .setVerdict(solution.getVerdict().ordinal())
                .build();
    }

    public static ProtoNodeState mapNodeStateToProto(NodeState state) {
        return ProtoNodeState.newBuilder()
                .setNodeType(state.getType().ordinal())
                .setJavaType(state.getJavaType())
                .setLabel(state.getLabel())
                .setOriginalLabel(state.getOriginalLabel())
                .setPositionInParent(state.getPositionInParent())
                .build();
    }

    public static List<ProtoNodeState> mapNodeStateArrayToProto(NodeState[] state) {
        return Arrays.stream(state)
                .map(EntityToProtoUtils::mapNodeStateToProto)
                .collect(Collectors.toList());
    }

    public static ProtoNodeContext mapNodeContextToProto(CodeChange.NodeContext context) {
        return ProtoNodeContext.newBuilder()
                .setNode(mapNodeStateToProto(context.getNode()))
                .setParent(mapNodeStateToProto(context.getParent()))
                .setParentOfParent(mapNodeStateToProto(context.getParentOfParent()))
                .addAllUncles(mapNodeStateArrayToProto(context.getUncles()))
                .addAllBrothers(mapNodeStateArrayToProto(context.getBrothers()))
                .addAllChildren(mapNodeStateArrayToProto(context.getChildren()))
                .build();

    }

    public static ProtoAtomicChange mapCodeChangeToProto(CodeChange change) {
        return ProtoAtomicChange.newBuilder()
                .setChangeType(change.getChangeType().ordinal())
                .setOriginalContext(mapNodeContextToProto(change.getOriginalContext()))
                .setDestinationContext(mapNodeContextToProto(change.getDestinationContext()))
                .build();
    }

    public static ProtoChanges mapChangesToProto(Changes changes) {
        final List<ProtoAtomicChange> atomicChanges = changes.getChanges().stream()
                .map(EntityToProtoUtils::mapCodeChangeToProto)
                .collect(Collectors.toList());
        return ProtoChanges.newBuilder()
                .setSrc(mapSolutionToProto(changes.getOrigin()))
                .setDst(mapSolutionToProto(changes.getTarget()))
                .addAllChanges(atomicChanges)
                .build();
    }

    public static ProtoVectorFeatures mapVectorFeaturesToProto(double[] vector) {
        final ProtoVectorFeatures.Builder builder = ProtoVectorFeatures.newBuilder();
        for (var value : vector) {
            builder.addFeatures(value);
        }
        return builder.build();
    }

    public static ProtoVectorListFeatures mapVectorListFeaturesToProto(List<double[]> features) {
        final List<ProtoVectorFeatures> protoFeatures = features.stream()
                .map(EntityToProtoUtils::mapVectorFeaturesToProto)
                .collect(Collectors.toList());
        return ProtoVectorListFeatures.newBuilder()
                .addAllFeatures(protoFeatures)
                .build();
    }

    public static ProtoStringListFeatures mapStringListFeaturesToProto(List<String> features) {
        return ProtoStringListFeatures.newBuilder()
                .addAllFeatures(features)
                .build();
    }

    public static ProtoChangeListFeatures mapChangeListFeaturesToProto(List<CodeChange> features) {
        final List<ProtoAtomicChange> protoFeatures = features.stream()
                .map(EntityToProtoUtils::mapCodeChangeToProto)
                .collect(Collectors.toList());
        return ProtoChangeListFeatures.newBuilder()
                .addAllFeatures(protoFeatures)
                .build();
    }

    public static <F> void tryCastFeaturesList(List<?> features, Class<F> template,
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

    public static ProtoFeaturesWrapper mapFeaturesWrapperToProto(Wrapper<?, ?> wrapper) {
        final ProtoFeaturesWrapper.Builder builder = ProtoFeaturesWrapper.newBuilder();
        if (wrapper.getMeta() instanceof Solution) {
            builder.setSolution(mapSolutionToProto((Solution) wrapper.getMeta()));
        } else {
            throw new IllegalArgumentException("Unsupported meta information type!");
        }
        if (wrapper.getFeatures() instanceof double[]) {
            builder.setVectorFeatures(mapVectorFeaturesToProto((double[]) wrapper.getFeatures()));
        } else if (wrapper.getFeatures() instanceof List) {
            final List<?> features = (List<?>) wrapper.getFeatures();
            tryCastFeaturesList(features, String.class, x ->
                    builder.setStringListFeatures(mapStringListFeaturesToProto(x)));
            tryCastFeaturesList(features, CodeChange.class, x ->
                    builder.setChangeListFeatures(mapChangeListFeaturesToProto(x)));
            tryCastFeaturesList(features, double[].class, x ->
                    builder.setVectorListFeatures(mapVectorListFeaturesToProto(x)));
        }
        return builder.build();
    }

    public static ProtoFeaturesWrapper mapFeaturesWrapperToProto(Solution solution) {
        return ProtoFeaturesWrapper.newBuilder()
                .setSolution(mapSolutionToProto(solution))
                .build();
    }

    public static <F> ProtoCluster mapClusterToProto(Cluster<?> cluster) {
        final List<ProtoFeaturesWrapper> solutions;
        if (cluster.stream().map(Object::getClass).allMatch(Solution.class::equals)) {
            solutions = cluster.stream()
                    .map(Solution.class::cast)
                    .map(EntityToProtoUtils::mapFeaturesWrapperToProto)
                    .collect(Collectors.toList());
        } else if (cluster.stream().map(Object::getClass).allMatch(Wrapper.class::equals)) {
            solutions = cluster.stream()
                    .map(Wrapper.class::cast)
                    .map(EntityToProtoUtils::mapFeaturesWrapperToProto)
                    .collect(Collectors.toList());
        } else {
            throw new IllegalArgumentException("Unsupported cluster elements types");
        }
        return ProtoCluster.newBuilder()
                .addAllSolutions(solutions)
                .build();
    }


}
