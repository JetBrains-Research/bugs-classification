package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.CodeChange;

import java.io.Serializable;

public interface EncodingStrategy extends Serializable {
    long encode(CodeChange value);
}
