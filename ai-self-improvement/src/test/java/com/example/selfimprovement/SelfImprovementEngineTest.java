package com.example.selfimprovement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static com.example.selfimprovement.SelfImprovementEngine.*;
import static org.junit.jupiter.api.Assertions.*;

class SelfImprovementEngineTest {
    @TempDir
    Path tempDirectory;

    private List<EvaluationCase> evaluations() throws Exception {
        return SelfImprovementEngine.loadEvaluationCases(Path.of("data/evaluations.jsonl"));
    }

    private List<RecipeCase> recipes() throws Exception {
        return SelfImprovementEngine.loadRecipeCases(Path.of("data/recipe_evaluations.jsonl"));
    }

    private Config config() throws Exception {
        return SelfImprovementEngine.loadConfig(Path.of("config.json"));
    }

    @Test
    void improverSelectsCandidateUsingTrainingSplit() throws Exception {
        List<EvaluationCase> train = evaluations().stream()
                .filter(row -> row.split().equals("train"))
                .toList();

        ChangeProposal proposal = SelfImprovementEngine.proposeChange(train, "accuracy_first", config());

        assertNotNull(proposal);
        assertEquals("prompt_checklist", proposal.variant());
        assertEquals(1.0, proposal.trainingMetrics().accuracy());
        assertEquals(0.5, proposal.baselineTrainingMetrics().accuracy());
    }

    @Test
    void heldoutGateAcceptsImprovementBeyondBootstrapNoise() throws Exception {
        List<EvaluationCase> heldout = evaluations().stream()
                .filter(row -> row.split().equals("heldout"))
                .toList();

        ChangeGate gate = SelfImprovementEngine.gateChange(heldout, "prompt_checklist", config(), 17);

        assertTrue(gate.passed(), gate.reasons().toString());
        assertEquals(1.0, (double) gate.candidate().get("accuracy"));
        assertTrue(gate.pairedBootstrap().lowerBound() > 0);
    }

    @Test
    void heldoutGateRejectsChangeOutsideModelAllowlist() throws Exception {
        List<EvaluationCase> heldout = evaluations().stream()
                .filter(row -> row.split().equals("heldout"))
                .toList();
        Config base = config();
        Config noPermittedModels = new Config(
                List.of(),
                base.minimumAccuracy(),
                base.minimumCategoryAccuracy(),
                base.maximumCategoryRegression(),
                base.minimumImprovement(),
                base.maximumEvaluationBudgetUsd(),
                base.bootstrapSamples(),
                base.confidenceLevel(),
                base.initialImproverRecipe(),
                base.minimumRecipeScore());

        ChangeGate gate = SelfImprovementEngine.gateChange(heldout, "prompt_checklist", noPermittedModels, 17);

        assertFalse(gate.passed());
        assertTrue(gate.reasons().contains("candidate model is not on the permitted-model list"));
    }

    @Test
    void recursiveImproverChoosesRecipeFromMetaTrainingAndGatesOnMetaHeldout() throws Exception {
        List<RecipeCase> cases = recipes();
        List<RecipeCase> metaTrain = cases.stream().filter(row -> row.split().equals("meta_train")).toList();
        List<RecipeCase> metaHeldout = cases.stream().filter(row -> row.split().equals("meta_heldout")).toList();
        Config config = config();

        RecipeProposal proposal = SelfImprovementEngine.proposeRecipe(metaTrain, "accuracy_first");
        assertNotNull(proposal);
        assertEquals("business_aware", proposal.proposedRecipe());

        RecipeGate gate = SelfImprovementEngine.gateRecipe(
                metaHeldout,
                proposal.currentRecipe(),
                proposal.proposedRecipe(),
                config,
                18);
        assertTrue(gate.passed(), gate.reasons().toString());
        assertTrue(gate.pairedBootstrap().lowerBound() > 0);
    }

    @Test
    void commandLineWritesLocalRunReport() throws Exception {
        int exitCode = Main.execute(new String[] {
                "--approve",
                "--output-dir",
                tempDirectory.toString()
        });

        assertEquals(0, exitCode);
        List<Path> reports;
        try (var files = java.nio.file.Files.list(tempDirectory)) {
            reports = files.filter(path -> path.getFileName().toString().startsWith("improve-")).toList();
        }
        assertEquals(1, reports.size());
        JsonNode report = new ObjectMapper().readTree(reports.getFirst().toFile());
        assertEquals("fully local deterministic simulation", report.path("mode").asText());
        assertEquals("kept", report.path("improvement").path("status").asText());
        assertTrue(report.path("improvement").path("human_approved").asBoolean());
        assertEquals("kept", report.path("recursive_recipe_tuning").path("status").asText());
        assertEquals("business_aware", report.path("active_improver_recipe").asText());
    }
}
