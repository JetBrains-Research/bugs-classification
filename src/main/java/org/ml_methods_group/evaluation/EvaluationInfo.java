package org.ml_methods_group.evaluation;

import java.nio.file.Path;
import java.nio.file.Paths;

public class EvaluationInfo {
    public static final Path PATH_TO_DATASET = Paths.get(".cache", "datasets");
    public static final Path PATH_TO_CLUSTERS = Paths.get(".cache", "clusters");
    public static final Path PATH_TO_CACHE = Paths.get(".cache", "cache");
    public static final Path PATH_TO_RESULTS = Paths.get(".cache", "results");
    public static final Path PATH_TO_PYTHON_BINARY = Paths.get(
            System.getenv().containsKey("PYTHONPATH")
                    ? System.getenv("PYTHONPATH")
                    : "/usr/bin/python3");
    public static final Path PATH_TO_PYTHON_SCRIPTS = Paths.get("python");
}
