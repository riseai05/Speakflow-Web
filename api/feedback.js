// Vercel serverless function. Runs on Node's built-in runtime — no npm
// install needed, `fetch` is globally available in Node 18+.
// Keeps GEMINI_API_KEY server-side; the browser never sees it.

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "Server missing GEMINI_API_KEY" });
    return;
  }

  const { durationSec, blinkCount, blinkRatePerMin, gazeStabilityScore, postureStabilityScore } =
    req.body || {};

  const isShortSession = durationSec < 25;
  const extremeMetrics = [gazeStabilityScore, postureStabilityScore].some((v) => v === 0 || v === 100);

  const prompt = `You are giving feedback to someone practicing answering interview questions on camera. Here is their session data:

- Duration: ${durationSec} seconds
- Blinks: ${blinkCount} total (${blinkRatePerMin} per minute)
- Gaze stability score: ${gazeStabilityScore}/100
- Posture stability score: ${postureStabilityScore}/100

Rules you must follow:
- Reference the specific numbers above directly — never give generic advice that could apply to anyone.
${isShortSession ? "- This session was very short (under 25 seconds) — explicitly acknowledge the sample is too small to draw a strong conclusion, rather than inventing confident feedback from thin data.\n" : ""}${extremeMetrics ? "- One or more scores hit an extreme (0 or 100) — flag that this likely reflects a measurement edge case or very brief/unusual sample, not necessarily a real pattern.\n" : ""}- Frame feedback around what these signals typically mean in an interview context. For example: gaze instability late in an answer often maps to losing confidence or running out of prepared points. Rigid, completely unchanging posture the whole time isn't necessarily good either — it can read as tense rather than composed.
- Keep it to 3-4 sentences total: one specific observation tied to their numbers, one likely interpretation of what that signal suggests in an interview context, one concrete suggestion for next time.
- Never use stock phrases like "make more eye contact" or "sit up straight" without tying them directly to the numbers above.
- Plain text only, no markdown formatting.`;

  try {
    const geminiRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: { maxOutputTokens: 400 },
        }),
      }
    );
    const geminiJson = await geminiRes.json();

    if (!geminiRes.ok) {
      console.error("Gemini API error:", JSON.stringify(geminiJson));
      res.status(200).json({
        feedback: "Coach feedback isn't available right now, but your session stats above are accurate — try again shortly.",
      });
      return;
    }

    const feedback =
      geminiJson.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ||
      "Your session data looked reasonable, but we couldn't generate written feedback this time — try again in a moment.";

    res.status(200).json({ feedback });
  } catch (err) {
    res.status(500).json({ error: "Failed to reach Gemini", detail: String(err) });
  }
};
