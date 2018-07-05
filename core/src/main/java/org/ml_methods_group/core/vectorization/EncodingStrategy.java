package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.AtomicChange;

import java.io.Serializable;

public interface EncodingStrategy extends Serializable {
    long encode(AtomicChange value);
    ChangeCodeWrapper decode(long code);
}
