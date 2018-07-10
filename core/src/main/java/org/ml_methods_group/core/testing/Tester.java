package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.List;

public interface Tester {
    double test(List<List<Wrapper>> clusters);
}
