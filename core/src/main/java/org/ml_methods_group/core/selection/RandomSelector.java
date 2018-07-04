package org.ml_methods_group.core.selection;

import java.util.*;

public class RandomSelector<T> implements RepresenterSelector<T> {

    final Random random = new Random(239566);

    @Override
    public List<T> findRepresenter(int n, List<T> samples) {
        final Set<T> result = new HashSet<>();
        while (result.size() < n) {
            result.add(samples.get(random.nextInt(samples.size())));
        }
        return new ArrayList<>(result);
    }
}
