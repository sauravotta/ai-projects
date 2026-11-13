package com.example.selfimprovement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

public final class SelfImprovementEngine {
    public static final String BASELINE = "baseline";
    public static final List<String> RECIPES = List.of("accuracy_first", "business_aware");
    public static final Map<String, CandidateSpec> CANDIDATES = Map.of(
            "prompt_checklist", new CandidateSpec(
                    "prompt",
                    "Add a short verification checklist to the prompt.",
                    "local-demo-small",
                    "low"),
            "model_upgrade", new CandidateSpec(
                    "model",
                    "Switch to the stronger permitted local simulator model.",
                    "local-demo-strong",
                    "low"),
            "reasoning_medium", new CandidateSpec(
                    "reasoning_effort",
                    "Raise reasoning effort from low to medium.",
                    "local-demo-small",
                    "medium"));

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private SelfImprovementEngine() {
    }

    public record Outcome(String label, double costUsd, double latencyMs) {
    }

    public record EvaluationCase(
            String id,
            String split,
            String category,
            String expected,
            double businessImpactUsd,
            Map<String, Outcome> outcomes) {
    }

    public record RecipeCase(String id, String split, Map<String, Double> recipeScores) {
    }

    public record Config(
            List<String> permittedModels,
            double minimumAccuracy,
            double minimumCategoryAccuracy,
            double maximumCategoryRegression,
            double minimumImprovement,
            double maximumEvaluationBudgetUsd,
            int bootstrapSamples,
            double confidenceLevel,
            String initialImproverRecipe,
            double minimumRecipeScore) {
    }

    public record CandidateSpec(String changeType, String description, String model, String reasoningEffort) {
    }

    public record Metrics(
            int caseCount,
            double accuracy,
            double businessMistakeCostUsd,
            double simulatedInferenceCostUsd,
            double meanLatencyMs,
            Map<String, Double> categoryAccuracy,
            List<Double> correctness) {
        public Map<String, Object> toReport() {
            Map<String, Object> report = new LinkedHashMap<>();
            report.put("case_count", caseCount);
            report.put("accuracy", accuracy);
            report.put("business_mistake_cost_usd", businessMistakeCostUsd);
            report.put("simulated_inference_cost_usd", simulatedInferenceCostUsd);
            report.put("mean_latency_ms", meanLatencyMs);
            report.put("category_accuracy", categoryAccuracy);
            return report;
        }
    }

    public record BootstrapDelta(
            double observedDelta,
            double confidenceLevel,
            double lowerBound,
            double upperBound,
            int bootstrapSamples) {
    }

    public record ChangeProposal(
            String variant,
            CandidateSpec change,
            double trainingScore,
            double trainingDelta,
            Metrics trainingMetrics,
            Metrics baselineTrainingMetrics,
            String recipe,
            String rationale) {
    }

    public record ChangeGate(
            Map<String, Object> baseline,
            Map<String, Object> candidate,
            BootstrapDelta pairedBootstrap,
            boolean passed,
            List<String> reasons) {
    }

    public record RecipeProposal(
            String currentRecipe,
            String proposedRecipe,
            Map<String, Double> metaTrainMeanScores,
            double metaTrainDelta,
            String rationale) {
    }

    public record RecipeGate(
            double currentRecipeMeanScore,
            double proposedRecipeMeanScore,
            BootstrapDelta pairedBootstrap,
            boolean passed,
            List<String> reasons) {
    }

    public static List<EvaluationCase> loadEvaluationCases(Path path) throws IOException {
        List<EvaluationCase> cases = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int lineNumber = 0;
        for (String line : Files.readAllLines(path)) {
            lineNumber++;
            if (line.isBlank()) {
                continue;
            }
            JsonNode row = parseObject(path, lineNumber, line);
            String id = requiredText(row, "id", path, lineNumber);
            if (!seen.add(id)) {
                throw invalid(path, lineNumber, "duplicate id " + id);
            }
            Map<String, Outcome> outcomes = new LinkedHashMap<>();
            JsonNode outcomeNodes = row.path("outcomes");
            if (!outcomeNodes.isObject()) {
                throw invalid(path, lineNumber, "outcomes must be an object");
            }
            outcomeNodes.fields().forEachRemaining(entry -> {
                JsonNode outcome = entry.getValue();
                outcomes.put(entry.getKey(), new Outcome(
                        outcome.path("label").isTextual() ? outcome.path("label").asText() : null,
                        numericValue(outcome.get("cost_usd")),
                        numericValue(outcome.get("latency_ms"))));
            });
            cases.add(new EvaluationCase(
                    id,
                    requiredText(row, "split", path, lineNumber),
                    requiredText(row, "category", path, lineNumber),
                    requiredText(row, "expected", path, lineNumber),
                    numericValue(row.get("business_impact_usd")),
                    outcomes));
        }
        if (cases.isEmpty()) {
            throw new IllegalArgumentException(path + " contains no evaluation cases");
        }
        return cases;
    }

    public static List<RecipeCase> loadRecipeCases(Path path) throws IOException {
        List<RecipeCase> cases = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int lineNumber = 0;
        for (String line : Files.readAllLines(path)) {
            lineNumber++;
            if (line.isBlank()) {
                continue;
            }
            JsonNode row = parseObject(path, lineNumber, line);
            String id = requiredText(row, "id", path, lineNumber);
            if (!seen.add(id)) {
                throw invalid(path, lineNumber, "duplicate id " + id);
            }
            JsonNode scoreNodes = row.path("recipe_scores");
            if (!scoreNodes.isObject()) {
                throw invalid(path, lineNumber, "recipe_scores must be an object");
            }
            Map<String, Double> scores = new LinkedHashMap<>();
            scoreNodes.fields().forEachRemaining(entry ->
                    scores.put(entry.getKey(), numericValue(entry.getValue())));
            cases.add(new RecipeCase(id, requiredText(row, "split", path, lineNumber), scores));
        }
        if (cases.isEmpty()) {
            throw new IllegalArgumentException(path + " contains no recipe evaluation cases");
        }
        return cases;
    }

    public static Config loadConfig(Path path) throws IOException {
        JsonNode node = MAPPER.readTree(path.toFile());
        if (node == null || !node.isObject()) {
            throw new IllegalArgumentException(path + ": configuration must be a JSON object");
        }
        List<String> permittedModels = new ArrayList<>();
        JsonNode models = node.path("permitted_models");
        if (!models.isArray()) {
            throw new IllegalArgumentException(path + ": permitted_models must be a JSON array");
        }
        models.forEach(model -> {
            if (!model.isTextual() || model.asText().isBlank()) {
                throw new IllegalArgumentException(path + ": permitted_models entries must be non-empty strings");
            }
            permittedModels.add(model.asText());
        });
        JsonNode bootstrapSamples = node.path("bootstrap_samples");
        return new Config(
                permittedModels,
                number(node, "minimum_accuracy"),
                number(node, "minimum_category_accuracy"),
                number(node, "maximum_category_regression"),
                number(node, "minimum_improvement"),
                number(node, "maximum_evaluation_budget_usd"),
                bootstrapSamples.isIntegralNumber() && bootstrapSamples.canConvertToInt()
                        ? bootstrapSamples.asInt()
                        : -1,
                number(node, "confidence_level"),
                node.path("initial_improver_recipe").asText(null),
                number(node, "minimum_recipe_score"));
    }

    private static JsonNode parseObject(Path path, int lineNumber, String line) {
        try {
            JsonNode node = MAPPER.readTree(line);
            if (node == null || !node.isObject()) {
                throw invalid(path, lineNumber, "each row must be a JSON object");
            }
            return node;
        } catch (IOException e) {
            throw invalid(path, lineNumber, "invalid JSON: " + e.getMessage());
        }
    }

    private static String requiredText(JsonNode node, String field, Path path, int lineNumber) {
        JsonNode value = node.get(field);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw invalid(path, lineNumber, field + " must be a non-empty string");
        }
        return value.asText();
    }

    private static double number(JsonNode node, String field) {
        return numericValue(node.get(field));
    }

    private static double numericValue(JsonNode value) {
        return value != null && value.isNumber() ? value.doubleValue() : Double.NaN;
    }

    private static IllegalArgumentException invalid(Path path, int lineNumber, String message) {
        return new IllegalArgumentException(path + ":" + lineNumber + ": " + message);
    }

    public static void validateInputs(
            List<EvaluationCase> cases,
            List<RecipeCase> recipeCases,
            Config config) {
        requireSplit(cases.stream().map(EvaluationCase::split).toList(), "train", "evaluation");
        requireSplit(cases.stream().map(EvaluationCase::split).toList(), "heldout", "evaluation");
        for (EvaluationCase evaluationCase : cases) {
            if (!Set.of("train", "heldout").contains(evaluationCase.split())) {
                throw new IllegalArgumentException(
                        "Case " + evaluationCase.id() + ": split must be train or heldout");
            }
            if (!Set.of("pass", "fail").contains(evaluationCase.expected())) {
                throw new IllegalArgumentException(
                        "Case " + evaluationCase.id() + ": expected must be pass or fail");
            }
            requireFiniteNonNegative(
                    evaluationCase.businessImpactUsd(),
                    "Case " + evaluationCase.id() + ": business_impact_usd");
            if (!evaluationCase.outcomes().containsKey(BASELINE)) {
                throw new IllegalArgumentException(
                        "Case " + evaluationCase.id() + ": outcomes must include baseline");
            }
            evaluationCase.outcomes().forEach((variant, outcome) -> {
                if (outcome.label() == null || !Set.of("pass", "fail").contains(outcome.label())) {
                    throw new IllegalArgumentException(
                            "Case " + evaluationCase.id() + ": " + variant + " label must be pass or fail");
                }
                requireFiniteNonNegative(outcome.costUsd(), "Case " + evaluationCase.id() + ": cost_usd");
                requireFiniteNonNegative(outcome.latencyMs(), "Case " + evaluationCase.id() + ": latency_ms");
            });
        }

        requireSplit(recipeCases.stream().map(RecipeCase::split).toList(), "meta_train", "recipe");
        requireSplit(recipeCases.stream().map(RecipeCase::split).toList(), "meta_heldout", "recipe");
        for (RecipeCase recipeCase : recipeCases) {
            if (!Set.of("meta_train", "meta_heldout").contains(recipeCase.split())) {
                throw new IllegalArgumentException(
                        "Recipe case " + recipeCase.id() + ": split must be meta_train or meta_heldout");
            }
            for (String recipe : RECIPES) {
                double score = recipeCase.recipeScores().getOrDefault(recipe, Double.NaN);
                if (!Double.isFinite(score) || score < 0 || score > 1) {
                    throw new IllegalArgumentException(
                            "Recipe case " + recipeCase.id() + ": " + recipe + " score must be between 0 and 1");
                }
            }
        }

        requireRange(config.minimumAccuracy(), "minimum_accuracy");
        requireRange(config.minimumCategoryAccuracy(), "minimum_category_accuracy");
        requireRange(config.maximumCategoryRegression(), "maximum_category_regression");
        requireRange(config.minimumImprovement(), "minimum_improvement");
        if (!Double.isFinite(config.confidenceLevel())
                || config.confidenceLevel() <= 0
                || config.confidenceLevel() >= 1) {
            throw new IllegalArgumentException("confidence_level must be greater than 0 and less than 1");
        }
        requireRange(config.minimumRecipeScore(), "minimum_recipe_score");
        requireFiniteNonNegative(config.maximumEvaluationBudgetUsd(), "maximum_evaluation_budget_usd");
        if (config.bootstrapSamples() < 100 || config.bootstrapSamples() > 1_000_000) {
            throw new IllegalArgumentException("bootstrap_samples must be an integer from 100 through 1000000");
        }
        if (!RECIPES.contains(config.initialImproverRecipe())) {
            throw new IllegalArgumentException("initial_improver_recipe must be one of " + RECIPES);
        }
        if (config.permittedModels().stream().anyMatch(String::isBlank)) {
            throw new IllegalArgumentException("permitted_models cannot contain blank model names");
        }
    }

    private static void requireSplit(List<String> splits, String required, String type) {
        if (!splits.contains(required)) {
            throw new IllegalArgumentException("The " + type + " dataset needs at least one " + required + " case");
        }
    }

    private static void requireRange(double value, String name) {
        if (!Double.isFinite(value) || value < 0 || value > 1) {
            throw new IllegalArgumentException(name + " must be between 0 and 1");
        }
    }

    private static void requireFiniteNonNegative(double value, String name) {
        if (!Double.isFinite(value) || value < 0) {
            throw new IllegalArgumentException(name + " must be finite and non-negative");
        }
    }

    public static Metrics evaluate(List<EvaluationCase> cases, String variant) {
        if (cases.isEmpty()) {
            throw new IllegalArgumentException("Cannot evaluate an empty case set");
        }
        List<Double> correctness = new ArrayList<>();
        Map<String, List<Boolean>> byCategory = new TreeMap<>();
        double mistakeCost = 0;
        double inferenceCost = 0;
        double latency = 0;
        for (EvaluationCase evaluationCase : cases) {
            Outcome outcome = evaluationCase.outcomes().get(variant);
            if (outcome == null) {
                throw new IllegalArgumentException(
                        "Variant " + variant + " is missing results for case " + evaluationCase.id());
            }
            boolean correct = outcome.label().equals(evaluationCase.expected());
            correctness.add(correct ? 1.0 : 0.0);
            byCategory.computeIfAbsent(evaluationCase.category(), ignored -> new ArrayList<>()).add(correct);
            if (!correct) {
                mistakeCost += evaluationCase.businessImpactUsd();
            }
            inferenceCost += outcome.costUsd();
            latency += outcome.latencyMs();
        }
        Map<String, Double> categoryAccuracy = new LinkedHashMap<>();
        byCategory.forEach((category, results) ->
                categoryAccuracy.put(category, results.stream().filter(Boolean::booleanValue).count() / (double) results.size()));
        return new Metrics(
                cases.size(),
                correctness.stream().mapToDouble(Double::doubleValue).average().orElseThrow(),
                mistakeCost,
                inferenceCost,
                latency / cases.size(),
                categoryAccuracy,
                List.copyOf(correctness));
    }

    public static double selectorScore(Metrics metrics, List<EvaluationCase> cases, String recipe) {
        if ("accuracy_first".equals(recipe)) {
            return metrics.accuracy();
        }
        if ("business_aware".equals(recipe)) {
            double totalImpact = cases.stream().mapToDouble(EvaluationCase::businessImpactUsd).sum();
            double normalizedMistakeCost = metrics.businessMistakeCostUsd() / Math.max(totalImpact, 1);
            return metrics.accuracy() - 0.5 * normalizedMistakeCost;
        }
        throw new IllegalArgumentException("Unknown improver recipe: " + recipe);
    }

    public static ChangeProposal proposeChange(
            List<EvaluationCase> trainingCases,
            String recipe,
            Config config) {
        return eligibleChangeProposals(trainingCases, recipe, config).stream()
                .max(Comparator.comparingDouble(ChangeProposal::trainingScore)
                        .thenComparingDouble(proposal -> -proposal.trainingMetrics().simulatedInferenceCostUsd())
                        .thenComparing(ChangeProposal::variant))
                .orElse(null);
    }

    public static List<ChangeProposal> eligibleChangeProposals(
            List<EvaluationCase> trainingCases,
            String recipe,
            Config config) {
        Metrics baselineMetrics = evaluate(trainingCases, BASELINE);
        double baselineScore = selectorScore(baselineMetrics, trainingCases, recipe);
        List<ChangeProposal> proposals = new ArrayList<>();
        for (Map.Entry<String, CandidateSpec> entry : CANDIDATES.entrySet()) {
            String variant = entry.getKey();
            CandidateSpec spec = entry.getValue();
            if (trainingCases.stream().anyMatch(row -> !row.outcomes().containsKey(variant))) {
                continue;
            }
            if (!config.permittedModels().contains(spec.model())) {
                continue;
            }
            Metrics metrics = evaluate(trainingCases, variant);
            if (metrics.simulatedInferenceCostUsd() > config.maximumEvaluationBudgetUsd()) {
                continue;
            }
            double score = selectorScore(metrics, trainingCases, recipe);
            if (score > baselineScore) {
                proposals.add(new ChangeProposal(
                        variant,
                        spec,
                        score,
                        score - baselineScore,
                        metrics,
                        baselineMetrics,
                        recipe,
                        "Highest-scoring eligible candidate under the configured offline selector."));
            }
        }
        return proposals.stream()
                .sorted(Comparator.comparingDouble(ChangeProposal::trainingScore).reversed()
                        .thenComparingDouble(proposal -> proposal.trainingMetrics().simulatedInferenceCostUsd())
                        .thenComparing(ChangeProposal::variant))
                .toList();
    }

    public static ChangeProposal proposalForVariant(
            List<ChangeProposal> eligibleProposals,
            String variant,
            String rationale) {
        return eligibleProposals.stream()
                .filter(proposal -> proposal.variant().equals(variant))
                .findFirst()
                .map(proposal -> new ChangeProposal(
                        proposal.variant(),
                        proposal.change(),
                        proposal.trainingScore(),
                        proposal.trainingDelta(),
                        proposal.trainingMetrics(),
                        proposal.baselineTrainingMetrics(),
                        proposal.recipe(),
                        rationale))
                .orElseThrow(() -> new IllegalArgumentException(
                        "Improver selected a candidate that is not eligible on training data: " + variant));
    }

    public static BootstrapDelta bootstrapDelta(
            List<Double> baselineValues,
            List<Double> candidateValues,
            int samples,
            double confidenceLevel,
            long seed) {
        if (baselineValues.isEmpty() || baselineValues.size() != candidateValues.size()) {
            throw new IllegalArgumentException("Bootstrap inputs must have equal, non-zero lengths");
        }
        List<Double> deltas = new ArrayList<>();
        for (int i = 0; i < baselineValues.size(); i++) {
            deltas.add(candidateValues.get(i) - baselineValues.get(i));
        }
        double observed = deltas.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        Random random = new Random(seed);
        double[] means = new double[samples];
        for (int sample = 0; sample < samples; sample++) {
            double sum = 0;
            for (int item = 0; item < deltas.size(); item++) {
                sum += deltas.get(random.nextInt(deltas.size()));
            }
            means[sample] = sum / deltas.size();
        }
        java.util.Arrays.sort(means);
        double tail = (1 - confidenceLevel) / 2;
        return new BootstrapDelta(
                observed,
                confidenceLevel,
                percentile(means, tail),
                percentile(means, 1 - tail),
                samples);
    }

    private static double percentile(double[] values, double quantile) {
        double position = (values.length - 1) * quantile;
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) {
            return values[lower];
        }
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    public static ChangeGate gateChange(
            List<EvaluationCase> heldoutCases,
            String variant,
            Config config,
            long seed) {
        Metrics baseline = evaluate(heldoutCases, BASELINE);
        Metrics candidate = evaluate(heldoutCases, variant);
        BootstrapDelta comparison = bootstrapDelta(
                baseline.correctness(),
                candidate.correctness(),
                config.bootstrapSamples(),
                config.confidenceLevel(),
                seed);
        List<String> reasons = new ArrayList<>();
        if (candidate.accuracy() < config.minimumAccuracy()) {
            reasons.add("overall accuracy is below its configured floor");
        }
        if (candidate.simulatedInferenceCostUsd() > config.maximumEvaluationBudgetUsd()) {
            reasons.add("evaluation cost exceeds the configured budget");
        }
        CandidateSpec spec = CANDIDATES.get(variant);
        if (spec == null || !config.permittedModels().contains(spec.model())) {
            reasons.add("candidate model is not on the permitted-model list");
        }
        if (comparison.observedDelta() < config.minimumImprovement()) {
            reasons.add("observed improvement is below the configured minimum");
        }
        if (comparison.lowerBound() <= 0) {
            reasons.add("paired bootstrap interval does not show improvement beyond zero");
        }
        candidate.categoryAccuracy().forEach((category, accuracy) -> {
            if (accuracy < config.minimumCategoryAccuracy()) {
                reasons.add(category + " accuracy is below its configured floor");
            }
            double baselineAccuracy = baseline.categoryAccuracy().get(category);
            if (baselineAccuracy - accuracy > config.maximumCategoryRegression()) {
                reasons.add(category + " regresses beyond the configured allowance");
            }
        });
        return new ChangeGate(
                baseline.toReport(),
                candidate.toReport(),
                comparison,
                reasons.isEmpty(),
                List.copyOf(reasons));
    }

    public static RecipeProposal proposeRecipe(List<RecipeCase> metaTrainingCases, String currentRecipe) {
        Map<String, Double> means = recipeMeans(metaTrainingCases);
        String proposed = RECIPES.stream()
                .max(Comparator.comparingDouble((String recipe) -> means.get(recipe))
                        .thenComparing(Comparator.naturalOrder()))
                .orElseThrow();
        if (proposed.equals(currentRecipe) || means.get(proposed) <= means.get(currentRecipe)) {
            return null;
        }
        return recipeProposalForChoice(
                metaTrainingCases,
                currentRecipe,
                proposed,
                "Highest-scoring recipe on meta-training data.");
    }

    public static Map<String, Double> recipeMeans(List<RecipeCase> metaTrainingCases) {
        Map<String, Double> means = new LinkedHashMap<>();
        for (String recipe : RECIPES) {
            double mean = metaTrainingCases.stream()
                    .mapToDouble(row -> row.recipeScores().get(recipe))
                    .average()
                    .orElseThrow();
            means.put(recipe, mean);
        }
        return Map.copyOf(means);
    }

    public static RecipeProposal recipeProposalForChoice(
            List<RecipeCase> metaTrainingCases,
            String currentRecipe,
            String proposed,
            String rationale) {
        if (!RECIPES.contains(proposed)) {
            throw new IllegalArgumentException("Improver selected an unknown recipe: " + proposed);
        }
        Map<String, Double> means = recipeMeans(metaTrainingCases);
        if (proposed.equals(currentRecipe) || means.get(proposed) <= means.get(currentRecipe)) {
            throw new IllegalArgumentException(
                    "Improver selected a recipe that does not outperform the current recipe on meta-training data");
        }
        return new RecipeProposal(
                currentRecipe,
                proposed,
                means,
                means.get(proposed) - means.get(currentRecipe),
                rationale);
    }

    public static RecipeGate gateRecipe(
            List<RecipeCase> metaHeldoutCases,
            String currentRecipe,
            String proposedRecipe,
            Config config,
            long seed) {
        List<Double> before = metaHeldoutCases.stream()
                .map(row -> row.recipeScores().get(currentRecipe))
                .toList();
        List<Double> after = metaHeldoutCases.stream()
                .map(row -> row.recipeScores().get(proposedRecipe))
                .toList();
        BootstrapDelta comparison = bootstrapDelta(
                before,
                after,
                config.bootstrapSamples(),
                config.confidenceLevel(),
                seed);
        double beforeMean = before.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        double afterMean = after.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        List<String> reasons = new ArrayList<>();
        if (afterMean < config.minimumRecipeScore()) {
            reasons.add("proposed recipe score is below its configured floor");
        }
        if (comparison.observedDelta() < config.minimumImprovement()) {
            reasons.add("recipe improvement is below the configured minimum");
        }
        if (comparison.lowerBound() <= 0) {
            reasons.add("paired bootstrap interval does not show recipe improvement beyond zero");
        }
        return new RecipeGate(
                beforeMean,
                afterMean,
                comparison,
                reasons.isEmpty(),
                List.copyOf(reasons));
    }
}
