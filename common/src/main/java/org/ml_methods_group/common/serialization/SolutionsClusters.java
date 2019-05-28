package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.Solution;

import java.io.*;
import java.nio.file.Path;
import java.util.List;

public class SolutionsClusters extends Clusters<Solution> {
    public SolutionsClusters(List<Cluster<Solution>> clusters) {
        super(clusters);
    }

    public SolutionsClusters(Clusters<Solution> clusters) {
        super(clusters.getClusters());
    }

    public static SolutionsClusters load(Path path) throws IOException {
        return JavaSerializationUtils.loadObject(SolutionsClusters.class, path);
    }

    public void store(Path path) throws IOException {
        JavaSerializationUtils.storeObject(this, path);
    }
}
