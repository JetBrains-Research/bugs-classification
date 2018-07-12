package org.ml_methods_group.core.changes;

import org.ml_methods_group.core.database.ConditionSupplier;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;

import java.util.ArrayList;
import java.util.List;

public class LazyChangeGenerator implements ChangeGenerator {

    private final Repository<CodeChange> repository;
    private final ConditionSupplier conditions;
    private final ChangeGenerator builder;

    public LazyChangeGenerator(Repository<CodeChange> repository, ChangeGenerator builder) {
        this.repository = repository;
        this.builder = builder;
        this.conditions = repository.conditionSupplier();
    }

    @Override
    public List<CodeChange> getChanges(Solution origin, Solution target) {
        final List<CodeChange> cache = tryLoad(origin.getSolutionId(), target.getSolutionId());
        if (!cache.isEmpty()) {
            return cache;
        }
        final List<CodeChange> result = builder.getChanges(origin, target);
        result.forEach(repository::insert);
        return result;
    }

    private List<CodeChange> tryLoad(int origin, int target) {
        final List<CodeChange> result = new ArrayList<>();
        repository.get(conditions.is("originSolutionId", origin),
                conditions.is("targetSolutionId", target))
                .forEachRemaining(result::add);
        return result;
    }
}
