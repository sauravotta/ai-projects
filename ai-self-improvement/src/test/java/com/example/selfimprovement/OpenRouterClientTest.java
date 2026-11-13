package com.example.selfimprovement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static com.example.selfimprovement.SelfImprovementEngine.*;
import static org.junit.jupiter.api.Assertions.*;

class OpenRouterClientTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void sendsOnlyEligibleTrainingCandidatesAndParsesOpenRouterResponse() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<String> authorization = new AtomicReference<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/chat/completions", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            byte[] body = """
                    {"model":"test/model","choices":[{"message":{"content":"{\\"choice\\":\\"prompt_checklist\\",\\"rationale\\":\\"It improves training quality within budget.\\"}"}}],"usage":{"prompt_tokens":120,"completion_tokens":18,"cost":0.0002}}
                    """.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            List<EvaluationCase> train = SelfImprovementEngine.loadEvaluationCases(Path.of("data/evaluations.jsonl"))
                    .stream()
                    .filter(row -> row.split().equals("train"))
                    .toList();
            List<ChangeProposal> eligible =
                    SelfImprovementEngine.eligibleChangeProposals(train, "accuracy_first", loadConfig());
            OpenRouterClient client = new OpenRouterClient(
                    "test-secret",
                    "test/model",
                    URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/chat/completions"),
                    HttpClient.newHttpClient());

            OpenRouterClient.Recommendation recommendation =
                    client.chooseCandidate(eligible, "accuracy_first");

            assertEquals("prompt_checklist", recommendation.choice());
            assertEquals("It improves training quality within budget.", recommendation.rationale());
            assertEquals("test/model", recommendation.usage().model());
            assertEquals(120, recommendation.usage().promptTokens());
            assertEquals(0.0002, recommendation.usage().reportedCostUsd());
            assertEquals("Bearer test-secret", authorization.get());
            JsonNode sent = MAPPER.readTree(requestBody.get());
            assertEquals("test/model", sent.path("model").asText());
            String prompt = sent.path("messages").path(1).path("content").asText();
            assertTrue(prompt.contains("prompt_checklist"));
            assertFalse(prompt.contains("heldout-format-01"));
            assertFalse(prompt.contains("meta-heldout"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void rejectsRecommendationsOutsideTheEligibleAllowlist() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/chat/completions", exchange -> {
            byte[] body = """
                    {"choices":[{"message":{"content":"{\\"choice\\":\\"arbitrary-code\\",\\"rationale\\":\\"Run generated code.\\"}"}}]}
                    """.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            List<EvaluationCase> train = SelfImprovementEngine.loadEvaluationCases(Path.of("data/evaluations.jsonl"))
                    .stream()
                    .filter(row -> row.split().equals("train"))
                    .toList();
            List<ChangeProposal> eligible =
                    SelfImprovementEngine.eligibleChangeProposals(train, "accuracy_first", loadConfig());
            OpenRouterClient client = new OpenRouterClient(
                    "test-secret",
                    "test/model",
                    URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/chat/completions"),
                    HttpClient.newHttpClient());

            IOException error = assertThrows(IOException.class, () -> client.chooseCandidate(eligible, "accuracy_first"));

            assertTrue(error.getMessage().contains("unknown or disallowed choice"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void recursiveImproverReceivesMetaTrainingScoresAndEligibleRecipesOnly() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/chat/completions", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] body = """
                    {"choices":[{"message":{"content":"{\\"choice\\":\\"business_aware\\",\\"rationale\\":\\"It scores higher on meta-training.\\"}"}}]}
                    """.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            OpenRouterClient client = new OpenRouterClient(
                    "test-secret",
                    "test/model",
                    URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/chat/completions"),
                    HttpClient.newHttpClient());

            OpenRouterClient.Recommendation recommendation = client.chooseRecipe(
                    "accuracy_first",
                    java.util.Map.of("accuracy_first", 0.73, "business_aware", 0.85),
                    List.of("business_aware"));

            assertEquals("business_aware", recommendation.choice());
            JsonNode sent = MAPPER.readTree(requestBody.get());
            String prompt = sent.path("messages").path(1).path("content").asText();
            assertTrue(prompt.contains("meta-training"));
            assertTrue(prompt.contains("business_aware"));
            assertFalse(prompt.contains("meta-heldout-01"));
            assertFalse(prompt.contains("0.89"));
        } finally {
            server.stop(0);
        }
    }

    private Config loadConfig() throws Exception {
        return SelfImprovementEngine.loadConfig(Path.of("config.json"));
    }
}
