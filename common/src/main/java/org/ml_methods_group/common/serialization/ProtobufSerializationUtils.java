package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.proto.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Function;

public class ProtobufSerializationUtils {

    //safe API

    public static void storeSolutionClusters(Clusters<Solution> clusters, Path path) throws IOException {
        final Path directory = path.getParent();
        if (directory != null && !Files.exists(directory) && !directory.toFile().mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transform(clusters).writeTo(outputStream);
        }
    }

    public static Clusters<Solution> loadSolutionClusters(Path path) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoClusters.parseFrom(inputStream));
        }
    }

    public static void storeChangesClusters(Clusters<Changes> clusters, Path path) throws IOException {
        final Path directory = path.getParent();
        if (directory != null && !Files.exists(directory) && !directory.toFile().mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transformChangesClusters(clusters).writeTo(outputStream);
        }
    }

    public static Clusters<Changes> loadChangesClusters(Path path) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoChangesClusters.parseFrom(inputStream));
        }
    }

    public static void storeMarkedClusters(MarkedClusters<Solution, String> clusters, Path path) throws IOException {
        final File directory = path.getParent().toFile();
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transform(clusters).writeTo(outputStream);
        }
    }

    public static MarkedClusters<Solution, String> loadMarkedClusters(Path path)
            throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoMarkedClusters.parseFrom(inputStream));
        }
    }

    public static void storeMarkedChangesClusters(MarkedClusters<Changes, String> clusters, Path path) throws IOException {
        final Path directory = path.getParent();
        if (directory != null && !Files.exists(directory) && !directory.toFile().mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transformMarkedChangesClusters(clusters).writeTo(outputStream);
        }
    }

    public static MarkedClusters<Changes, String> loadMarkedChangesClusters(Path path) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoMarkedChangesClusters.parseFrom(inputStream));
        }
    }

    public static void storeSolutionMarksHolder(SolutionMarksHolder holder, Path path) throws IOException {
        final File directory = path.getParent().toFile();
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transform(holder).writeTo(outputStream);
        }
    }

    public static SolutionMarksHolder loadSolutionMarksHolder(Path path) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoSolutionMarksHolder.parseFrom(inputStream));
        }
    }

    public static void storeDataset(Dataset dataset, Path path) throws IOException {
        final File directory = path.getParent().toFile();
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
            EntityToProtoUtils.transform(dataset).writeTo(outputStream);
        }
    }

    public static Dataset loadDataset(Path path) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transform(ProtoDataset.parseFrom(inputStream));
        }
    }
}
