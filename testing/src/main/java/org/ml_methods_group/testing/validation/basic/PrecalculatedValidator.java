package org.ml_methods_group.testing.validation.basic;

import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.SerializationUtils;
import org.ml_methods_group.testing.validation.Validator;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class PrecalculatedValidator implements Validator<Solution, String>, Serializable {

    private final Map<Integer, Set<String>> validMarks;

    public PrecalculatedValidator(Map<Integer, Set<String>> validMarks) {
        this.validMarks = validMarks;
    }

    @Override
    public boolean isValid(Solution value, String mark) {
        final Set<String> marks = validMarks.get(value.getSolutionId());
        if (marks == null) {
            throw new IllegalArgumentException();
        }
        return marks.contains(mark);
    }

    public void store(Path path) throws IOException {
        SerializationUtils.storeObject(this, path);
    }

    public static PrecalculatedValidator load(Path path) throws IOException {
        return SerializationUtils.loadObject(PrecalculatedValidator.class, path);
    }

    public static PrecalculatedValidator create(List<Solution> solutions) {
        try (Scanner scanner = new Scanner(System.in)) {
            final Map<Integer, Set<String>> result = new HashMap<>();
            boolean addMode = true;
            for (Solution solution : solutions) {
                final Set<String> marks = result.computeIfAbsent(solution.getSolutionId(), x -> new HashSet<>());
                if (!marks.isEmpty()) continue;
                System.out.println("Start marking solution(id=" + solution.getSolutionId() + "):");
                System.out.println(solution.getCode());
                System.out.println();
                label:
                while (true) {
                    System.out.println("Current marks: " + marks.stream().collect(Collectors.joining(" ")));
                    System.out.print(addMode ? "Add mark: " : "Remove mark: ");
                    final String mark = scanner.next();
                    switch (mark) {
                        case "=":
                            break label;
                        case "-":
                            addMode = false;
                            continue label;
                        case "+":
                            addMode = true;
                            continue label;
                    }
                    if (addMode) {
                        marks.add(mark);
                    } else {
                        marks.remove(mark);
                    }
                }
                System.out.println();
            }
            return new PrecalculatedValidator(result);
        }
    }
}
