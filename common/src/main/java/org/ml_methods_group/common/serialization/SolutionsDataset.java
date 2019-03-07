package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;

import java.io.*;
import java.nio.file.Path;
import java.util.Collection;
import java.util.function.Predicate;

public class SolutionsDataset extends Dataset<Solution> implements Serializable {
    public SolutionsDataset(Collection<Solution> values) {
        super(values);
    }

    @Override
    public SolutionsDataset filter(Predicate<Solution> predicate) {
        return new SolutionsDataset(getValues(predicate));
    }

    public static SolutionsDataset load(Path path) throws IOException {
        return JavaSerializationUtils.loadObject(SolutionsDataset.class, path);
    }

    public void store(Path path) throws IOException {
        JavaSerializationUtils.storeObject(this, path);
    }
}
