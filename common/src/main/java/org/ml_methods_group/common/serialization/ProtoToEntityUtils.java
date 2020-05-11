package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.proto.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

    public static Cluster<Solution> transform(ProtoCluster proto) {
        final List<Solution> solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return new Cluster<>(solutions);
    }

    public static Clusters<Solution> transform(ProtoClusters clusters) {
        final var clustersList = clusters.getClustersList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return new Clusters<>(clustersList);
    }

    public static Cluster<Changes> transform(ProtoChangesCluster proto) {
        final var solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return new Cluster<>(solutions);
    }

    public static Clusters<Changes> transform(ProtoChangesClusters clusters) {
        final var clustersList = clusters.getClustersList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return new Clusters<>(clustersList);
    }

    public static Map.Entry<Cluster<Solution>, String> transform(ProtoMarkedCluster proto) {
        final List<Solution> solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return Map.entry(new Cluster<>(solutions), proto.getMark());
    }

    public static MarkedClusters<Solution, String> transform(ProtoMarkedClusters proto) {
        final Map<Cluster<Solution>, String> map = new HashMap<>();
        proto.getClustersList().stream()
                .map(ProtoToEntityUtils::transform)
                .forEach(e -> map.put(e.getKey(), e.getValue()));
        return new MarkedClusters<>(map);
    }

    public static Map.Entry<Cluster<Changes>, String> transform(ProtoMarkedChangesCluster proto) {
        final List<Changes> solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return Map.entry(new Cluster<>(solutions), proto.getMark());
    }

    public static MarkedClusters<Changes, String> transform(ProtoMarkedChangesClusters proto) {
        final Map<Cluster<Changes>, String> map = new HashMap<>();
        proto.getClustersList().stream()
                .map(ProtoToEntityUtils::transform)
                .forEach(e -> map.put(e.getKey(), e.getValue()));
        return new MarkedClusters<>(map);
    }

    public static SolutionMarksHolder transform(ProtoSolutionMarksHolder proto) {
        final HashMap<Solution, List<String>> buffer = new HashMap<>();
        for (var solutionMarks : proto.getMapList()) {
            buffer.put(transform(solutionMarks.getSolution()),
                    new ArrayList<>(solutionMarks.getMarksList()));
        }
        return new SolutionMarksHolder(buffer);
    }

    public static Dataset transform(ProtoDataset proto) {
        final List<Solution> solutions = proto.getSolutionsList().stream()
                .map(ProtoToEntityUtils::transform)
                .collect(Collectors.toList());
        return new Dataset(solutions);
    }
}
