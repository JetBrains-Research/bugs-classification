package org.ml_methods_group.parsing;

import java.util.Optional;

public interface CodeValidator {
    Optional<String> validate(String code);
}
