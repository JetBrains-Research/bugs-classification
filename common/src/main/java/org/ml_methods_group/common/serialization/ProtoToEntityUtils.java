package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.Wrapper;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.proto.*;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ProtoToEntityUtils {
    public static Solution mapProtoToSolution(ProtoSolution proto) {
        return new Solution(
                proto.getCode(),
                proto.getProblemId(),
                proto.getSessionId(),
                proto.getSolutionId(),
                Solution.Verdict.valueOf(proto.getVerdict())
        );
    }

    public static CodeChange.NodeState mapProtoToNodeState(ProtoNodeState proto) {
        return new CodeChange.NodeState(
                NodeType.valueOf(proto.getNodeType()),
                proto.getJavaType(),
                proto.getLabel(),
                proto.getOriginalLabel(),
                proto.getPositionInParent()
        );
    }

    public static CodeChange.NodeState[] mapProtoToNodeStateArray(List<ProtoNodeState> protos) {
        return protos.stream().map(ProtoToEntityUtils::mapProtoToNodeState).toArray(CodeChange.NodeState[]::new);
    }

    public static CodeChange.NodeContext mapProtoToNodeContext(ProtoNodeContext proto) {
        return new CodeChange.NodeContext(
                mapProtoToNodeState(proto.getNode()),
                mapProtoToNodeState(proto.getParent()),
                mapProtoToNodeState(proto.getParentOfParent()),
                mapProtoToNodeStateArray(proto.getUnclesList()),
                mapProtoToNodeStateArray(proto.getBrothersList()),
                mapProtoToNodeStateArray(proto.getChildrenList())
        );
    }

    public static CodeChange mapProtoToCodeChange(ProtoAtomicChange proto) {
        return new CodeChange(
                mapProtoToNodeContext(proto.getOriginalContext()),
                mapProtoToNodeContext(proto.getDestinationContext()),
                ChangeType.valueOf(proto.getChangeType())
        );
    }

    public static Changes mapProtoToChanges(ProtoChanges proto) {
        return new Changes(
                mapProtoToSolution(proto.getSrc()),
                mapProtoToSolution(proto.getDst()),
                proto.getChangesList().stream()
                        .map(ProtoToEntityUtils::mapProtoToCodeChange)
                        .collect(Collectors.toList())
        );
    }

    public static double[] mapProtoToVectorFeatures(ProtoVectorFeatures proto) {
        final double[] vector = new double[proto.getFeaturesCount()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = proto.getFeatures(i);
        }
        return vector;
    }

    public static Wrapper<List<double[]>, Solution> extractVectorListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasVectorListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                proto.getVectorListFeatures().getFeaturesList().stream()
                        .map(ProtoToEntityUtils::mapProtoToVectorFeatures)
                        .collect(Collectors.toList()),
                mapProtoToSolution(proto.getSolution())
        );
    }

    public static Wrapper<List<String>, Solution> extractStringListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasStringListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                proto.getStringListFeatures().getFeaturesList(),
                mapProtoToSolution(proto.getSolution())
        );
    }

    public static Wrapper<List<CodeChange>, Solution> extractChangeListFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasChangeListFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                proto.getChangeListFeatures().getFeaturesList().stream()
                        .map(ProtoToEntityUtils::mapProtoToCodeChange)
                        .collect(Collectors.toList()),
                mapProtoToSolution(proto.getSolution())
        );
    }

    public static Wrapper<double[], Solution> extractVectorFeatures(ProtoFeaturesWrapper proto) {
        if (!proto.hasVectorFeatures()) {
            throw new IllegalArgumentException("Such features type wasn't provided!");
        }
        return new Wrapper<>(
                mapProtoToVectorFeatures(proto.getVectorFeatures()),
                mapProtoToSolution(proto.getSolution())
        );
    }

    public static Solution extractSolutions(ProtoFeaturesWrapper proto) {
        return mapProtoToSolution(proto.getSolution());
    }

    public static <F> Cluster<F> mapCluster(ProtoCluster proto,
                                            Function<ProtoFeaturesWrapper, F> extractor) {
        final List<F> solutions = proto.getSolutionsList().stream()
                .map(extractor)
                .collect(Collectors.toList());
        return new Cluster<>(solutions);
    }

    public static <F> Map.Entry<Cluster<F>, String> mapMarkedCluster(ProtoMarkedCluster proto,
                                                                     Function<ProtoFeaturesWrapper, F> extractor) {
        final List<F> solutions = proto.getSolutionsList().stream()
                .map(extractor)
                .collect(Collectors.toList());
        return Map.entry(new Cluster<>(solutions), proto.getMark());
    }
}
