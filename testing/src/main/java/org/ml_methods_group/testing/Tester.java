package org.ml_methods_group.testing;


import org.ml_methods_group.common.Wrapper;

import java.util.List;

public interface Tester<F, M> {
    TestingResults test(List<List<Wrapper<F, M>>> clusters);
}
