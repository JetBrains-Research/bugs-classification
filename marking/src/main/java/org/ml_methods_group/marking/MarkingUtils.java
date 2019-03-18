package org.ml_methods_group.marking;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.marking.markers.ManualClusterMarker;

import java.util.HashMap;
import java.util.Map;

public class MarkingUtils {
    public static MarkedSolutionsClusters markClusters(Clusters<Solution> clusters, int bound) {
        final ManualClusterMarker marker = new ManualClusterMarker(System.in, System.out, 5);
        final Map<Cluster<Solution>, String> marks = new HashMap<>();
        clusters.getClusters().stream()
                .filter(cluster -> cluster.size() >= bound)
                .forEach(cluster -> {
                    final String mark = marker.mark(cluster);
                    if (mark != null) {
                        marks.put(cluster, mark);
                    }
                });
        return new MarkedSolutionsClusters(marks);
    }
}
