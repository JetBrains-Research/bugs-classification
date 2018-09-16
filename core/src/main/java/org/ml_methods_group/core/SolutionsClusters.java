package org.ml_methods_group.core;

import org.ml_methods_group.core.entities.Solution;

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
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (SolutionsClusters) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Wrong file format!", e);
        }
    }

    public void store(Path path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this);
        }
    }
}
