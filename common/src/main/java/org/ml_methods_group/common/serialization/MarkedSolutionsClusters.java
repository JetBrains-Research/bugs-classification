package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.MarkedClusters;
import org.ml_methods_group.common.Solution;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

public class MarkedSolutionsClusters extends MarkedClusters<Solution, String> {

    public MarkedSolutionsClusters(Map<Cluster<Solution>, String> marks) {
        super(marks);
    }


    public static MarkedSolutionsClusters load(Path path) throws IOException {
        return SerializationUtils.loadObject(MarkedSolutionsClusters.class, path);
    }

    public void store(Path path) throws IOException {
        SerializationUtils.storeObject(this, path);
    }
}
