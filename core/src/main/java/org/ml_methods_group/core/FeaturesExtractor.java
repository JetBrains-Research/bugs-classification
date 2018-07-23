package org.ml_methods_group.core;

import java.io.*;

public interface FeaturesExtractor<T, F> extends Serializable {
    F process(T value);
}
