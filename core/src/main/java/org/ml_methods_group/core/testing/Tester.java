package org.ml_methods_group.core.testing;


import org.ml_methods_group.core.Wrapper;

import java.util.List;

public interface Tester<F, M> {
    TestingResults test(List<List<Wrapper<F, M>>> clusters);
}
