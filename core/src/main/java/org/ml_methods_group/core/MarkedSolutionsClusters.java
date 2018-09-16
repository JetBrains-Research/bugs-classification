package org.ml_methods_group.core;

import org.ml_methods_group.core.entities.Solution;

import java.io.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class MarkedSolutionsClusters extends MarkedClusters<Solution, String> {

    public MarkedSolutionsClusters(Map<Cluster<Solution>, String> marks) {
        super(marks);
    }

    public static MarkedSolutionsClusters load(Path path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (MarkedSolutionsClusters) ois.readObject();
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
