package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.proto.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ProtoToEntityUtils {
    public static Solution transform(ProtoSolution proto) {
        return new Solution(
                proto.getCode(),
                proto.getProblemId(),
                proto.getSessionId(),
                proto.getSolutionId(),
                Solution.Verdict.valueOf(proto.getVerdict())
        );
    }

    public static CodeChange.NodeState transform(ProtoNodeState proto) {
        return new CodeChange.NodeState(
                NodeType.valueOf(proto.getNodeType()),
                proto.getJavaType(),
                proto.getLabel(),
                proto.getOriginalLabel(),
                proto.getPositionInParent()
        );
    }

    public static CodeChange.NodeState[] transform(List<ProtoNodeState> protos) {
        return protos.stream().map(ProtoToEntityUtils::transform).toArray(CodeChange.NodeState[]::new);
    }

    public static CodeChange.NodeContext transform(ProtoNodeContext proto) {
        return new CodeChange.NodeContext(
                transform(proto.getNode()),
                transform(proto.getParent()),
                transform(proto.getParentOfParent()),
                transform(proto.getUnclesList()),
                transform(proto.getBrothersList()),
                transform(proto.getChildrenList())
        );
    }

    public static CodeChange transform(ProtoAtomicChange proto) {
        return new CodeChange(
                transform(proto.getOriginalContext()),
                transform(proto.getDestinationContext()),
                ChangeType.valueOf(proto.getChangeType())
        );
    }

    public static Changes transform(ProtoChanges proto) {
        return new Changes(
                transform(proto.getSrc()),
                transform(proto.getDst()),
                proto.getChangesList().stream()
                        .map(ProtoToEntityUtils::transform)
                        .collect(Collectors.toList())
        );
    }

    public static double[] transform(ProtoVectorFeatures proto) {
        final double[] vector = new double[proto.getFeaturesCount()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = proto.getFeatures(i);
        }
        return vector;
    }

    public static List<double[]> transform(ProtoVectorListFeatures proto) {
        return proto.getFeaturesList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
    }

    public static List<String> transform(ProtoStringListFeatures proto) {
        return proto.getFeaturesList();
    }

    public static List<CodeChange> transform(ProtoChangeListFeatures proto) {
        return proto.getFeaturesList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
    }

    public static Wrapper<List<double[]>, Solution> extractVectorListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasVectorListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                transform(proto.getVectorListFeatures()),
                transform(proto.getSolution())
        );
    }

    public static Wrapper<List<String>, Solution> extractStringListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasStringListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                transform(proto.getStringListFeatures()),
                transform(proto.getSolution())
        );
    }

    public static Wrapper<List<CodeChange>, Solution> extractChangeListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasChangeListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                transform(proto.getChangeListFeatures()),
                transform(proto.getSolution())
        );
    }

    public static Wrapper<double[], Solution> extractVectorFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasVectorFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                transform(proto.getVectorFeatures()),
                transform(proto.getSolution())
        );
    }

    public static Solution extractSolutions(ProtoFeaturesWrapper proto) {
        return transform(proto.getSolution());
    }

    public static <F> Cluster<F> transformCluster(ProtoCluster proto,
                                                  Function<ProtoFeaturesWrapper, F> extractor) {
        final List<F> solutions = proto.getSolutionsList().stream()
                .map(extractor)
                .collect(Collectors.toList());
        return new Cluster<>(solutions);
    }

    public static <F> Clusters<F> transformClusters(ProtoClusters clusters,
                                                    Function<ProtoFeaturesWrapper, F> extractor) {
        final var clustersList = clusters.getClustersList().stream()
                .map(cluster -> transformCluster(cluster, extractor))
                .collect(Collectors.toList());
        return new Clusters<>(clustersList);
    }

    public static Map.Entry<Cluster<Solution>, String> transformMarkedCluster(ProtoMarkedCluster proto) {
        final List<Solution> solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return Map.entry(new Cluster<>(solutions), proto.getMark());
    }

    public static MarkedClusters<Solution, String> transformMarkedClusters(ProtoMarkedClusters proto) {
        final Map<Cluster<Solution>, String> map = new HashMap<>();
        proto.getClustersList().stream()
                .map(ProtoToEntityUtils::transformMarkedCluster)
                .forEach(e -> map.put(e.getKey(), e.getValue()));
        return new MarkedClusters<>(map);
    }
}
