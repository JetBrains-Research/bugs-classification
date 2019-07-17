package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.proto.*;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
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

//    public static ProtoFeaturesWrapper transform(Wrapper<?, ?> wrapper) {
//        final ProtoFeaturesWrapper.Builder builder = ProtoFeaturesWrapper.newBuilder();
//        if (wrapper.getMeta() instanceof Solution) {
//            builder.setSolution(transform((Solution) wrapper.getMeta()));
//        } else {
//            throw new IllegalArgumentException("Unsupported meta information type!");
//        }
//        if (wrapper.getFeatures() instanceof double[]) {
//            builder.setVectorFeatures(transformVectorFeatures((double[]) wrapper.getFeatures()));
//        } else if (wrapper.getFeatures() instanceof List) {
//            final List<?> features = (List<?>) wrapper.getFeatures();
//            tryCastFeaturesList(features, String.class, x ->
//                    builder.setStringListFeatures(transformStringListFeatures(x)));
//            tryCastFeaturesList(features, CodeChange.class, x ->
//                    builder.setChangeListFeatures(transformChangeListFeatures(x)));
//            tryCastFeaturesList(features, double[].class, x ->
//                    builder.setVectorListFeatures(transformVectorListFeatures(x)));
//        }
//        return builder.build();
//    }


    public static ProtoCluster transformSolutionsCluster(Cluster<Solution> cluster) {
        final List<ProtoSolution> solutions = cluster.stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoCluster.newBuilder()
                .addAllSolutions(solutions)
                .build();
    }

    public static ProtoClusters transform(Clusters<Solution> clusters) {
        final List<ProtoCluster> proto = clusters.getClusters()
                .stream()
                .map(EntityToProtoUtils::transformSolutionsCluster)
                .collect(Collectors.toList());
        return ProtoClusters.newBuilder()
                .addAllClusters(proto)
                .build();
    }

    public static ProtoMarkedChangesCluster transform(Map.Entry<Cluster<Changes>, String> cluster) {
        final var protos = cluster.getKey().stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoMarkedChangesCluster.newBuilder()
                .addAllSolutions(protos)
                .setMark(cluster.getValue())
                .build();
    }

    public static ProtoMarkedChangesClusters transformMarkedChangesClusters(MarkedClusters<Changes, String> clusters) {
        final List<ProtoMarkedChangesCluster> proto = clusters.getMarks()
                .entrySet()
                .stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoMarkedChangesClusters.newBuilder()
                .addAllClusters(proto)
                .build();
    }

    public static ProtoChangesCluster transformChangesCluster(Cluster<Changes> cluster) {
        final List<ProtoChanges> solutions = cluster.stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoChangesCluster.newBuilder()
                .addAllSolutions(solutions)
                .build();
    }

    public static ProtoChangesClusters transformChangesClusters(Clusters<Changes> clusters) {
        final List<ProtoChangesCluster> proto = clusters.getClusters()
                .stream()
                .map(EntityToProtoUtils::transformChangesCluster)
                .collect(Collectors.toList());
        return ProtoChangesClusters.newBuilder()
                .addAllClusters(proto)
                .build();
    }

    public static ProtoMarkedCluster transform(Cluster<Solution> cluster, String mark) {
        final List<ProtoSolution> proto = cluster.stream()
                .map(EntityToProtoUtils::transform)
                .collect(Collectors.toList());
        return ProtoMarkedCluster.newBuilder()
                .addAllSolutions(proto)
                .setMark(mark)
                .build();
    }

    public static ProtoMarkedClusters transform(MarkedClusters<Solution, String> clusters) {
        final List<ProtoMarkedCluster> proto = clusters.getMarks().entrySet()
                .stream()
                .map(e -> transform(e.getKey(), e.getValue()))
                .collect(Collectors.toList());
        return ProtoMarkedClusters.newBuilder()
                .addAllClusters(proto)
                .build();
    }

    public static ProtoSolutionMarksHolder transform(SolutionMarksHolder holder) {
        final var builder = ProtoSolutionMarksHolder.newBuilder();
        for (var entry : holder) {
            final var solutionMarks = ProtoSolutionsMarks.newBuilder()
                    .addAllMarks(entry.getValue())
                    .setSolution(transform(entry.getKey()))
                    .build();
            builder.addMap(solutionMarks);
        }
        return builder.build();
    }

    public static ProtoDataset transform(Dataset dataset) {
        final var builder = ProtoDataset.newBuilder();
        dataset.forEach(x -> builder.addSolutions(transform(x)));
        return builder.build();
    }
}
