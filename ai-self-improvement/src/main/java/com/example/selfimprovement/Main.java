package com.example.selfimprovement;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

import static com.example.selfimprovement.SelfImprovementEngine.*;

public final class Main {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Path DEFAULT_DATASET = Path.of("data/evaluations.jsonl");
    private static final Path DEFAULT_RECIPE_DATASET = Path.of("data/recipe_evaluations.jsonl");
    private static final Path DEFAULT_CONFIG = Path.of("config.json");

    private Main() {
    }

    public static void main(String[] args) {
        int exitCode;
        try {
            exitCode = execute(args);
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            exitCode = 2;
        }
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static int execute(String[] args) throws IOException {
        Options options = parseOptions(args);
        if (options.help()) {
            printUsage();
            return 0;
        }
        List<EvaluationCase> cases = SelfImprovementEngine.loadEvaluationCases(options.dataset());
        List<RecipeCase> recipeCases = SelfImprovementEngine.loadRecipeCases(options.recipeDataset());
        Config config = SelfImprovementEngine.loadConfig(options.config());
        SelfImprovementEngine.validateInputs(cases, recipeCases, config);

        String initialRecipe = config.initialImproverRecipe();
        Map<String, Object> report = new LinkedHashMap<>();
        String runId = "improve-" + UUID.randomUUID().toString().replace("-", "").substring(0, 10);
        report.put("run_id", runId);
        report.put("created_at", Instant.now().toString());
        report.put("mode", "fully local deterministic simulation");
        report.put("active_improver_recipe", initialRecipe);

        List<EvaluationCase> train = cases.stream().filter(row -> row.split().equals("train")).toList();
        List<EvaluationCase> heldout = cases.stream().filter(row -> row.split().equals("heldout")).toList();
        List<RecipeCase> metaTrain = recipeCases.stream().filter(row -> row.split().equals("meta_train")).toList();
        List<RecipeCase> metaHeldout = recipeCases.stream().filter(row -> row.split().equals("meta_heldout")).toList();

        runAgentImprovement(report, train, heldout, config, options);
        runRecursiveTuning(report, metaTrain, metaHeldout, config, options);

        Files.createDirectories(options.outputDirectory());
        Path reportPath = options.outputDirectory().resolve(runId + ".json");
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(reportPath.toFile(), report);
        System.out.println("\nRun report: " + reportPath);
        return 0;
    }

    private static void runAgentImprovement(
            Map<String, Object> report,
            List<EvaluationCase> train,
            List<EvaluationCase> heldout,
            Config config,
            Options options) {
        ChangeProposal proposal = SelfImprovementEngine.proposeChange(train, config.initialImproverRecipe(), config);
        if (proposal == null) {
            report.put("improvement", Map.of(
                    "status", "no_proposal",
                    "reason", "No permitted candidate improves the training score within the configured budget."));
            System.out.println("Improver: no qualifying prompt, model, or reasoning-effort change found on training data.");
            return;
        }

        CandidateSpec spec = proposal.change();
        System.out.printf(
                Locale.ROOT,
                "\nImprover proposal: %s -> %s%n%s%nTraining accuracy: %.1f%% -> %.1f%% "
                        + "(selector score +%.3f; held-out data not used)%n",
                spec.changeType(),
                proposal.variant(),
                spec.description(),
                proposal.baselineTrainingMetrics().accuracy() * 100,
                proposal.trainingMetrics().accuracy() * 100,
                proposal.trainingDelta());

        boolean approved = askApproval("agent change " + proposal.variant(), options.autoApprove());
        if (!approved) {
            report.put("improvement", Map.of(
                    "status", "rejected_by_human",
                    "human_approved", false,
                    "proposal", proposal));
            System.out.println("Improver change not applied.");
            return;
        }

        ChangeGate gate = SelfImprovementEngine.gateChange(heldout, proposal.variant(), config, options.seed());
        String status = gate.passed() ? "kept" : "rolled_back";
        Map<String, Object> stage = new LinkedHashMap<>();
        stage.put("status", status);
        stage.put("human_approved", true);
        stage.put("proposal", proposal);
        stage.put("heldout_gate", gate);
        report.put("improvement", stage);
        System.out.printf(
                Locale.ROOT,
                "Held-out accuracy: %.1f%% -> %.1f%%; %.0f%% paired-bootstrap interval [%.3f, %.3f]%n",
                (double) gate.baseline().get("accuracy") * 100,
                (double) gate.candidate().get("accuracy") * 100,
                gate.pairedBootstrap().confidenceLevel() * 100,
                gate.pairedBootstrap().lowerBound(),
                gate.pairedBootstrap().upperBound());
        if (gate.passed()) {
            System.out.println("Promotion gate passed: keep the candidate change.");
        } else {
            System.out.println("Promotion gate failed: roll back to the baseline.");
            gate.reasons().forEach(reason -> System.out.println("  - " + reason));
        }
    }

    private static void runRecursiveTuning(
            Map<String, Object> report,
            List<RecipeCase> metaTrain,
            List<RecipeCase> metaHeldout,
            Config config,
            Options options) {
        String currentRecipe = config.initialImproverRecipe();
        RecipeProposal proposal = SelfImprovementEngine.proposeRecipe(metaTrain, currentRecipe);
        if (proposal == null) {
            report.put("recursive_recipe_tuning", Map.of(
                    "status", "no_proposal",
                    "reason", "No alternative improver recipe scored higher on meta-training data."));
            System.out.println("\nRecursive improver: no better recipe proposed on meta-training data.");
            return;
        }

        System.out.printf(
                "\nRecursive proposal: tune the improver recipe %s -> %s%n"
                        + "The meta-improver compares recipe choices on meta-training data; "
                        + "it has not read meta-held-out scores.%n",
                proposal.currentRecipe(),
                proposal.proposedRecipe());
        boolean approved = askApproval("recursive recipe change " + proposal.proposedRecipe(), options.autoApprove());
        if (!approved) {
            report.put("recursive_recipe_tuning", Map.of(
                    "status", "rejected_by_human",
                    "human_approved", false,
                    "proposal", proposal));
            System.out.println("Recursive recipe change not applied.");
            return;
        }

        RecipeGate gate = SelfImprovementEngine.gateRecipe(
                metaHeldout,
                proposal.currentRecipe(),
                proposal.proposedRecipe(),
                config,
                options.seed() + 1);
        String status = gate.passed() ? "kept" : "rolled_back";
        Map<String, Object> stage = new LinkedHashMap<>();
        stage.put("status", status);
        stage.put("human_approved", true);
        stage.put("proposal", proposal);
        stage.put("meta_heldout_gate", gate);
        report.put("recursive_recipe_tuning", stage);
        System.out.printf(
                Locale.ROOT,
                "Meta-held-out mean recipe score: %.3f -> %.3f; paired-bootstrap interval [%.3f, %.3f]%n",
                gate.currentRecipeMeanScore(),
                gate.proposedRecipeMeanScore(),
                gate.pairedBootstrap().lowerBound(),
                gate.pairedBootstrap().upperBound());
        if (gate.passed()) {
            report.put("active_improver_recipe", proposal.proposedRecipe());
            System.out.println("Recursive gate passed: keep " + proposal.proposedRecipe() + " as the improver recipe.");
        } else {
            System.out.println("Recursive gate failed: retain " + currentRecipe + ".");
            gate.reasons().forEach(reason -> System.out.println("  - " + reason));
        }
    }

    private static boolean askApproval(String stage, boolean autoApprove) {
        if (autoApprove) {
            System.out.println("Human approval: approved by --approve for " + stage + ".");
            return true;
        }
        java.io.Console console = System.console();
        if (console == null) {
            System.out.println("Human approval: rejected " + stage + " (non-interactive run; pass --approve to approve).");
            return false;
        }
        String response = console.readLine("Approve proposed %s? [y/N] ", stage);
        return response != null && List.of("y", "yes").contains(response.strip().toLowerCase(Locale.ROOT));
    }

    private static Options parseOptions(String[] args) {
        Path dataset = DEFAULT_DATASET;
        Path recipeDataset = DEFAULT_RECIPE_DATASET;
        Path config = DEFAULT_CONFIG;
        Path outputDirectory = Path.of("runs");
        long seed = 17;
        boolean autoApprove = false;
        boolean help = false;
        List<String> arguments = new ArrayList<>(List.of(args));
        for (int i = 0; i < arguments.size(); i++) {
            String argument = arguments.get(i);
            switch (argument) {
                case "--approve" -> autoApprove = true;
                case "--help", "-h" -> help = true;
                case "--dataset" -> dataset = Path.of(optionValue(arguments, ++i, argument));
                case "--recipe-dataset" -> recipeDataset = Path.of(optionValue(arguments, ++i, argument));
                case "--config" -> config = Path.of(optionValue(arguments, ++i, argument));
                case "--output-dir" -> outputDirectory = Path.of(optionValue(arguments, ++i, argument));
                case "--seed" -> {
                    try {
                        seed = Long.parseLong(optionValue(arguments, ++i, argument));
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("--seed must be an integer");
                    }
                }
                default -> throw new IllegalArgumentException("Unknown option: " + argument);
            }
        }
        return new Options(dataset, recipeDataset, config, outputDirectory, seed, autoApprove, help);
    }

    private static String optionValue(List<String> arguments, int index, String option) {
        if (index >= arguments.size() || arguments.get(index).startsWith("--")) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return arguments.get(index);
    }

    private static void printUsage() {
        System.out.println("""
                Usage: java -jar target/ai-self-improvement-1.0.0.jar [options]
                  --approve                 Explicitly approve both proposed changes
                  --dataset PATH            Simulated evaluation JSONL
                  --recipe-dataset PATH     Meta-evaluation JSONL
                  --config PATH             Guardrail JSON file
                  --output-dir PATH         Local report directory (default: runs)
                  --seed N                  Paired-bootstrap seed (default: 17)
                  --help                    Show this help
                """);
    }

    private record Options(
            Path dataset,
            Path recipeDataset,
            Path config,
            Path outputDirectory,
            long seed,
            boolean autoApprove,
            boolean help) {
    }
}
