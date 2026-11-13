package com.example.selfimprovement;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class MainOpenRouterIntegrationTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void runsBothOpenRouterRecommendationsThroughTheCliWorkflow(@TempDir Path tempDir) throws Exception {
        AtomicInteger callCount = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/chat/completions", exchange -> {
            exchange.getRequestBody().readAllBytes();
            int call = callCount.getAndIncrement();
            String recommendation = call == 0
                    ? "{\"choice\":\"prompt_checklist\",\"rationale\":\"Improves the eligible training score.\"}"
                    : "{\"choice\":\"business_aware\",\"rationale\":\"Improves the meta-training score.\"}";
            byte[] body = ("{\"model\":\"test/model\",\"choices\":[{\"message\":{\"content\":\""
                    + recommendation.replace("\"", "\\\"")
                    + "\"}}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8,\"cost\":0.0001}}")
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
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
            Path outputDirectory = tempDir.resolve("runs");

            assertEquals(0, Main.execute(
                    new String[]{"--llm", "--approve", "--output-dir", outputDirectory.toString()},
                    client));

            assertEquals(2, callCount.get());
            Path reportFile;
            try (var reportFiles = Files.list(outputDirectory)) {
                reportFile = reportFiles.findFirst().orElseThrow();
            }
            JsonNode report = MAPPER.readTree(reportFile.toFile());
            assertTrue(report.path("llm").path("enabled").asBoolean());
            assertEquals("OpenRouter", report.path("llm").path("provider").asText());
            assertEquals(2, report.path("llm").path("calls").size());
            assertEquals(
                    "candidate_improver",
                    report.path("llm").path("calls").get(0).path("task").asText());
            assertEquals(
                    "recursive_improver",
                    report.path("llm").path("calls").get(1).path("task").asText());
            assertEquals(
                    "prompt_checklist",
                    report.path("improvement").path("proposal").path("variant").asText());
            assertEquals(
                    "business_aware",
                    report.path("recursive_recipe_tuning")
                            .path("proposal")
                            .path("proposedRecipe")
                            .asText());
        } finally {
            server.stop(0);
        }
    }
}
