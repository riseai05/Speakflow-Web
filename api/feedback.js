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

  const prompt = `You are a calm, encouraging presence and communication coach. A user just finished a ${durationSec}-second practice session (speech, pitch, or interview answer) in front of their webcam. Here is what was measured during the session:

- Blinks: ${blinkCount} total (${blinkRatePerMin} per minute — typical relaxed resting rate is 15-20/min; higher often signals nervousness or eye strain)
- Gaze stability: ${gazeStabilityScore}/100 (higher means their eyes stayed steady/on-camera rather than darting around)
- Posture stability: ${postureStabilityScore}/100 (higher means their head position stayed steady rather than shifting/tilting)

Write a short (3-5 sentence) constructive feedback summary covering their focus, composure, and presence. Be specific about what the numbers suggest, encouraging in tone, and give one concrete tip they could try next time. Do not use markdown formatting, just plain text.`;

  try {
    const geminiRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`,
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
    const feedback =
      geminiJson.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ||
      "Your session data looked reasonable, but we couldn't generate written feedback this time — try again in a moment.";

    res.status(200).json({ feedback });
  } catch (err) {
    res.status(500).json({ error: "Failed to reach Gemini", detail: String(err) });
  }
};
