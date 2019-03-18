package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.Wrapper;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.proto.ProtoClusters;
import org.ml_methods_group.common.proto.ProtoFeaturesWrapper;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Function;

public class ProtobufSerializationUtils {

    //safe API

    public static void storeSolutionClusters(Clusters<Solution> clusters, Path path) throws IOException {
        unsafeStoreClusters(clusters, path);
    }

    public static void storeVectorFeaturesWrapperClusters(Clusters<Wrapper<double[], Solution>> clusters,
                                                          Path path) throws IOException {
        unsafeStoreClusters(clusters, path);
    }

    public static void storeVectorListFeaturesWrapperClusters(Clusters<Wrapper<List<double[]>, Solution>> clusters,
                                                          Path path) throws IOException {
        unsafeStoreClusters(clusters, path);
    }

    public static void storeStringListFeaturesWrapperClusters(Clusters<Wrapper<List<String>, Solution>> clusters,
                                                              Path path) throws IOException {
        unsafeStoreClusters(clusters, path);
    }

    public static void storeChangeListFeaturesWrapperClusters(Clusters<Wrapper<List<CodeChange>, Solution>> clusters,
                                                              Path path) throws IOException {
        unsafeStoreClusters(clusters, path);
    }

    public static Clusters<Solution> loadSolutionClusters(Path path) throws IOException {
        return unsafeLoadClusters(path, ProtoToEntityUtils::extractSolutions);
    }

    public static Clusters<Wrapper<double[], Solution>> loadVectorFeaturesWrapperClusters(Path path)
            throws IOException {
        return unsafeLoadClusters(path, ProtoToEntityUtils::extractVectorFeatures);
    }

    public static Clusters<Wrapper<List<double[]>, Solution>> loadVectorListFeaturesWrapperClusters(Path path)
            throws IOException {
        return unsafeLoadClusters(path, ProtoToEntityUtils::extractVectorListFeatures);
    }

    public static Clusters<Wrapper<List<String>, Solution>> loadStringListFeaturesWrapperClusters(Path path)
            throws IOException {
        return unsafeLoadClusters(path, ProtoToEntityUtils::extractStringListFeatures);
    }

    public static Clusters<Wrapper<List<CodeChange>, Solution>> loadChangeListFeaturesWrapperClusters(Path path)
            throws IOException {
        return unsafeLoadClusters(path, ProtoToEntityUtils::extractChangeListFeatures);
    }

    //unsafe methods

    private static void unsafeStoreClusters(Clusters<?> clusters, Path path) throws IOException {
        final File directory = path.getParent().toFile();
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream outputStream = new FileOutputStream(path.toFile())) {
                EntityToProtoUtils.transform(clusters).writeTo(outputStream);
        }
    }

    private static <T> Clusters<T> unsafeLoadClusters(Path path,
                                                      Function<ProtoFeaturesWrapper, T> extractor) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(path.toFile())) {
            return ProtoToEntityUtils.transformClusters(ProtoClusters.parseFrom(inputStream), extractor);
        }
    }
}