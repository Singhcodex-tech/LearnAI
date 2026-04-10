<?php
declare(strict_types=1);

/*
 * Lightweight PHP proxy for the Flask API.
 * Useful on hosts where a PHP front controller is preferred.
 *
 * Usage examples:
 * - POST /api.php?endpoint=generate
 * - POST /api.php?endpoint=quiz
 * - GET  /api.php?endpoint=stats&session_id=<id>
 *
 * Set FLASK_API_BASE in server env to override upstream.
 */

header("Content-Type: application/json; charset=utf-8");

$allowed = [
    "health",
    "generate",
    "quiz",
    "feedback",
    "reteach",
    "stats",
    "visual",
    "chat",
    "summary",
    "note",
    "sources",
    "generate_ppt",
];

$endpoint = isset($_GET["endpoint"]) ? strtolower(trim((string)$_GET["endpoint"])) : "";
if ($endpoint === "" || !in_array($endpoint, $allowed, true)) {
    http_response_code(400);
    echo json_encode(["error" => "Invalid or missing endpoint"]);
    exit;
}

$base = getenv("FLASK_API_BASE");
if (!$base) {
    $base = "http://127.0.0.1:5000";
}
$base = rtrim($base, "/");

$method = strtoupper($_SERVER["REQUEST_METHOD"] ?? "GET");

if ($endpoint === "stats") {
    $sessionId = isset($_GET["session_id"]) ? trim((string)$_GET["session_id"]) : "";
    if ($sessionId === "") {
        http_response_code(400);
        echo json_encode(["error" => "Missing session_id"]);
        exit;
    }
    $target = $base . "/stats/" . rawurlencode($sessionId);
} else {
    $target = $base . "/" . $endpoint;
}

$payload = file_get_contents("php://input");

$ch = curl_init($target);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);
curl_setopt($ch, CURLOPT_TIMEOUT, 60);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
if ($method !== "GET" && $payload !== false && $payload !== "") {
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
}

$response = curl_exec($ch);
$curlErr = curl_error($ch);
$status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
curl_close($ch);

if ($response === false) {
    http_response_code(502);
    echo json_encode(["error" => "Upstream request failed", "details" => $curlErr]);
    exit;
}

http_response_code($status > 0 ? $status : 200);
echo $response;