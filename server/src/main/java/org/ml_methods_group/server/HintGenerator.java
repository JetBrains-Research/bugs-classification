package org.ml_methods_group.server;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;

import javax.annotation.Resource;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;

@Path("/bugs-classification")
public class HintGenerator {

    public final Map<Integer, Classifier<Solution, String>> classifier;
//
    public HintGenerator() {
        try {
            this.data = ProtobufSerializationUtils.loadMarkedClusters(
                    Paths.get("C:\\internship\\bugs-classification\\.cache\\clusters\\factorial\\def_jac_0.2\\step_2\\40_clusters.tmp"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/hint")
    public String getHint(@QueryParam("problem") int problemId,
            @QueryParam("code") String code) {
//        final Solution solution = new Solution(code, problemId, -1, -1, FAIL);
//        final Optional<String> result = classifier.classify(solution);
//        return result.orElse("");
        return "result! " + problemId + " " + code + " " + data.getFlatMarks().values().iterator().next();
    }
}
