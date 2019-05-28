package org.ml_methods_group.common;

import javax.annotation.Nonnull;
import java.util.*;
import java.util.stream.Collectors;

public class SolutionMarksHolder implements Iterable<Map.Entry<Solution, List<String>>> {
    private final Map<Solution, List<String>> marks;

    public SolutionMarksHolder(Map<Solution, List<String>> marks) {
        this.marks = marks;
    }

    public SolutionMarksHolder() {
        this(new HashMap<>());
    }

    public void addMark(Solution solution, String mark) {
        marks.computeIfAbsent(solution, x -> new ArrayList<>()).add(mark);
    }

    public void clearMarks(Solution solution) {
        marks.remove(solution);
    }

    public List<Solution> getSolutions() {
        return marks.keySet().stream()
                .sorted(Comparator.comparingInt(Solution::getSolutionId))
                .collect(Collectors.toList());
    }

    public Optional<List<String>> getMarks(Solution solution) {
        return Optional.ofNullable(marks.get(solution))
                .map(Collections::unmodifiableList);
    }

    public int size() {
        return marks.size();
    }

    @Override
    @Nonnull
    public Iterator<Map.Entry<Solution, List<String>>> iterator() {
        final var iterator = marks.entrySet().iterator();
        return new Iterator<>() {
            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public Map.Entry<Solution, List<String>> next() {
                final var entry = iterator.next();
                return Map.entry(entry.getKey(), Collections.unmodifiableList(entry.getValue()));
            }
        };
    }
}
