package org.ml_methods_group.core;

import java.io.File;

public interface CSVParser {
    void parse(File file, SolutionDatabase solutions);
}
