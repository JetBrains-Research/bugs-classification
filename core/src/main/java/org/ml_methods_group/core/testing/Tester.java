package org.ml_methods_group.core.testing;


import org.ml_methods_group.core.Wrapper;

import java.util.List;

public interface Tester<F> {
    double test(List<List<Wrapper<F>>> clusters);
}
