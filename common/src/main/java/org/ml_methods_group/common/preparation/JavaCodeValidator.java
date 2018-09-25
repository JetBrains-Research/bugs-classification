package org.ml_methods_group.common.preparation;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;

import java.util.Optional;

public class JavaCodeValidator {

    public Optional<String> validate(String code) {
        if (checkValid(code)) {
            return Optional.of(code);
        }
        final String wrapped = "class MyMagicWrapper {\n" + code + "\n}";
        return checkValid(wrapped) ? Optional.of(wrapped) : Optional.empty();
    }

    private boolean checkValid(String code) {
        try {
            JavaParser.parse(code);
        } catch (ParseProblemException e) {
            return false;
        }
        return true;
    }
}
