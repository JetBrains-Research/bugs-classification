package org.ml_methods_group.parsing;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;

import java.util.Optional;

public interface CodeValidator {
    Optional<String> validate(String code);
}
