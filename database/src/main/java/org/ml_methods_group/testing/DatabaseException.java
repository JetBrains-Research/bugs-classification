package org.ml_methods_group.testing;

public class DatabaseException extends RuntimeException {
    DatabaseException(String message) {
        super(message);
    }

    DatabaseException(String message, Throwable cause) {
        super(message, cause);
    }

    DatabaseException(Throwable cause) {
        super(cause);
    }
}
