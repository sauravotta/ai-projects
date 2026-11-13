package com.example.selfimprovement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static com.example.selfimprovement.SelfImprovementEngine.*;

public final class OpenRouterClient {
    public static final String DEFAULT_MODEL = "openai/gpt-4o-mini";
    private static final URI DEFAULT_ENDPOINT = URI.create("https://openrouter.ai/api/v1/chat/completions");
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String apiKey;
    private final String model;
    private final URI endpoint;
    private final HttpClient httpClient;

    public OpenRouterClient(String apiKey, String model) {
        this(apiKey, model, DEFAULT_ENDPOINT, HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .build());
    }

    OpenRouterClient(String apiKey, String model, URI endpoint, HttpClient httpClient) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("OpenRouter API key must not be empty");
        }
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("OpenRouter model must not be empty");
        }
        this.apiKey = apiKey;
        this.model = model;
        this.endpoint = endpoint;
        this.httpClient = httpClient;
    }

    public String model() {
        return model;
    }

    public record Usage(String model, Integer promptTokens, Integer completionTokens, Double reportedCostUsd) {
        public Map<String, Object> toReport() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("model", model);
            result.put("prompt_tokens", promptTokens);
            result.put("completion_tokens", completionTokens);
            result.put("reported_cost_usd", reportedCostUsd);
            return result;
        }
    }

    public record Recommendation(String choice, String rationale, Usage usage) {
    }

    public Recommendation chooseCandidate(List<ChangeProposal> eligibleProposals, String recipe) throws IOException {
        if (eligibleProposals.isEmpty()) {
            throw new IllegalArgumentException("Cannot ask OpenRouter to choose from an empty candidate list");
        }
        List<Map<String, Object>> choices = new ArrayList<>();
        for (ChangeProposal proposal : eligibleProposals) {
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("variant", proposal.variant());
            choice.put("change_type", proposal.change().changeType());
            choice.put("description", proposal.change().description());
            choice.put("model", proposal.change().model());
            choice.put("reasoning_effort", proposal.change().reasoningEffort());
            choice.put("training_accuracy", proposal.trainingMetrics().accuracy());
            choice.put("training_selector_score", proposal.trainingScore());
            choice.put("training_score_delta", proposal.trainingDelta());
            choice.put("business_mistake_cost_usd", proposal.trainingMetrics().businessMistakeCostUsd());
            choice.put("simulated_cost_usd", proposal.trainingMetrics().simulatedInferenceCostUsd());
            choice.put("category_accuracy", proposal.trainingMetrics().categoryAccuracy());
            choice.put("baseline_training_accuracy", proposal.baselineTrainingMetrics().accuracy());
            choice.put(
                    "baseline_business_mistake_cost_usd",
                    proposal.baselineTrainingMetrics().businessMistakeCostUsd());
            choice.put("baseline_category_accuracy", proposal.baselineTrainingMetrics().categoryAccuracy());
            choices.add(choice);
        }
        String prompt = """
                Choose exactly one candidate improvement from this JSON list. These options were already checked
                against the training split, model allowlist, and configured budget. Do not invent candidates and
                do not request or infer held-out results. Prefer robust metric improvements and avoid costly or
                category-specific regressions. The current improver recipe is %s.

                Candidates:
                %s

                Return only a JSON object with a "choice" equal to one listed variant and a concise "rationale".
                """.formatted(recipe, MAPPER.writeValueAsString(choices));
        Set<String> eligibleChoices = eligibleProposals.stream()
                .map(ChangeProposal::variant)
                .collect(Collectors.toUnmodifiableSet());
        return requestRecommendation("candidate", prompt, eligibleChoices);
    }

    public Recommendation chooseRecipe(
            String currentRecipe,
            Map<String, Double> metaTrainingScores,
            List<String> eligibleRecipes)
            throws IOException {
        if (eligibleRecipes.isEmpty()) {
            throw new IllegalArgumentException("Cannot ask OpenRouter to choose from an empty recipe list");
        }
        String prompt = """
                You are selecting a recipe for a recursive self-improvement system.
                Choose only between the eligible recipes listed below using the meta-training mean scores.
                Do not invent recipes and do not request or infer meta-held-out scores.
                The current recipe is "%s".
                Eligible recipes: %s
                Meta-training scores:
                %s
                Return only a JSON object with a "choice" equal to one eligible recipe and a concise "rationale".
                """.formatted(currentRecipe, MAPPER.writeValueAsString(eligibleRecipes),
                MAPPER.writeValueAsString(metaTrainingScores));
        return requestRecommendation("recipe", prompt, Set.copyOf(eligibleRecipes));
    }

    private Recommendation requestRecommendation(String task, String prompt, Set<String> allowedChoices)
            throws IOException {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("model", model);
        body.put("temperature", 0.1);
        body.put("max_tokens", 300);
        ArrayNode messages = body.putArray("messages");
        messages.addObject()
                .put("role", "system")
                .put("content", "You are a cautious evaluation improver. Follow constraints and return valid JSON only.");
        messages.addObject().put("role", "user").put("content", prompt);

        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(Duration.ofSeconds(60))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .header("HTTP-Referer", "https://github.com/sauravotta/ai-projects")
                .header("X-Title", "AI Self-Improvement POC")
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("OpenRouter " + task + " request was interrupted", e);
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            String errorBody = response.body() == null ? "" : response.body();
            if (errorBody.length() > 500) {
                errorBody = errorBody.substring(0, 500) + "...";
            }
            throw new IOException(
                    "OpenRouter " + task + " request failed with HTTP " + response.statusCode()
                            + (errorBody.isBlank() ? "" : ": " + errorBody));
        }

        JsonNode payload;
        try {
            payload = MAPPER.readTree(response.body());
        } catch (IOException e) {
            throw new IOException("OpenRouter " + task + " response was not valid JSON", e);
        }
        if (payload == null || !payload.isObject()) {
            throw new IOException("OpenRouter " + task + " response must be a JSON object");
        }
        JsonNode content = payload.path("choices").path(0).path("message").path("content");
        if (!content.isTextual()) {
            throw new IOException("OpenRouter " + task + " response did not contain text content");
        }
        JsonNode verdict;
        try {
            verdict = MAPPER.readTree(stripCodeFences(content.asText()));
        } catch (IOException e) {
            throw new IOException("OpenRouter " + task + " response must contain a JSON recommendation", e);
        }
        String choice = verdict.path("choice").isTextual() ? verdict.path("choice").asText() : "";
        String rationale = verdict.path("rationale").isTextual() ? verdict.path("rationale").asText().strip() : "";
        if (!allowedChoices.contains(choice)) {
            throw new IOException("OpenRouter " + task + " selected an unknown or disallowed choice: " + choice);
        }
        if (rationale.isBlank()) {
            throw new IOException("OpenRouter " + task + " response must include a concise rationale");
        }
        return new Recommendation(choice, rationale, readUsage(payload));
    }

    private Usage readUsage(JsonNode payload) {
        JsonNode usage = payload.path("usage");
        JsonNode promptTokens = usage.get("prompt_tokens");
        JsonNode completionTokens = usage.get("completion_tokens");
        JsonNode cost = usage.get("cost");
        return new Usage(
                payload.path("model").isTextual() ? payload.path("model").asText() : model,
                promptTokens != null && promptTokens.canConvertToInt() ? promptTokens.asInt() : null,
                completionTokens != null && completionTokens.canConvertToInt() ? completionTokens.asInt() : null,
                cost != null && cost.isNumber() && Double.isFinite(cost.doubleValue()) && cost.doubleValue() >= 0
                        ? cost.doubleValue()
                        : null);
    }

    private String stripCodeFences(String content) {
        String trimmed = content.strip();
        if (!trimmed.startsWith("```")) {
            return trimmed;
        }
        int firstNewline = trimmed.indexOf('\n');
        int lastFence = trimmed.lastIndexOf("```");
        if (firstNewline < 0 || lastFence <= firstNewline) {
            return trimmed;
        }
        return trimmed.substring(firstNewline + 1, lastFence).strip();
    }
}
