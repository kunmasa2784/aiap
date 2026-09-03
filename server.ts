import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

function pcmToWav(pcmBuffer: Buffer, sampleRate = 24000, numChannels = 1): Buffer {
  const wavHeader = Buffer.alloc(44);
  const byteRate = sampleRate * numChannels * 2;
  const blockAlign = numChannels * 2;
  const dataSize = pcmBuffer.length;
  const fileSize = 36 + dataSize;

  wavHeader.write("RIFF", 0);
  wavHeader.writeUInt32LE(fileSize, 4);
  wavHeader.write("WAVE", 8);
  wavHeader.write("fmt ", 12);
  wavHeader.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
  wavHeader.writeUInt16LE(1, 20);  // AudioFormat (1 for PCM)
  wavHeader.writeUInt16LE(numChannels, 22);
  wavHeader.writeUInt32LE(sampleRate, 24);
  wavHeader.writeUInt32LE(byteRate, 28);
  wavHeader.writeUInt16LE(blockAlign, 32);
  wavHeader.writeUInt16LE(16, 34); // BitsPerSample
  wavHeader.write("data", 36);
  wavHeader.writeUInt32LE(dataSize, 40);

  return Buffer.concat([wavHeader, pcmBuffer]);
}

let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI | null {
  if (!aiClient && process.env.GEMINI_API_KEY) {
    aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return aiClient;
}

// In-memory cache for TTS audio buffers to accelerate playback and avoid re-synthesizing
const ttsServerCache = new Map<string, Buffer>();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "50mb" }));

  // API 1: Health check & config status
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      hasGeminiKey: !!process.env.GEMINI_API_KEY,
    });
  });

  // API 2: High-Fidelity Natural Flash TTS Narration
  app.post("/api/tts", async (req, res) => {
    try {
      const { text, voice = "Aoede" } = req.body;
      if (!text || typeof text !== "string" || !text.trim()) {
        return res.status(400).json({ error: "Text is required" });
      }

      const ai = getGenAI();
      if (!ai) {
        return res.status(503).json({
          error: "GEMINI_API_KEY is not configured",
          fallback: true,
        });
      }

      const validVoices = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"];
      const selectedVoice = validVoices.includes(voice) ? voice : "Aoede";
      const cleanText = text.trim();
      const cacheKey = `${selectedVoice}:${cleanText}`;

      if (ttsServerCache.has(cacheKey)) {
        res.setHeader("Content-Type", "audio/wav");
        res.setHeader("Cache-Control", "public, max-age=86400");
        return res.send(ttsServerCache.get(cacheKey));
      }

      const prompt = `落ち着いた映画紀行ドキュメンタリーのプロの朗読者として、情感を込めて優しく自然に語りかけてください：\n${cleanText}`;

      const aiResponse = await ai.models.generateContent({
        model: "gemini-3.1-flash-tts-preview",
        contents: prompt,
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: selectedVoice },
            },
          },
        },
      });

      const part = aiResponse.candidates?.[0]?.content?.parts?.[0];
      if (!part || !part.inlineData || !part.inlineData.data) {
        throw new Error("No audio data returned from Gemini TTS");
      }

      const pcmBuffer = Buffer.from(part.inlineData.data, "base64");
      const wavBuffer = pcmToWav(pcmBuffer, 24000, 1);
      ttsServerCache.set(cacheKey, wavBuffer);

      res.setHeader("Content-Type", "audio/wav");
      res.setHeader("Cache-Control", "public, max-age=86400");
      return res.send(wavBuffer);
    } catch (err: any) {
      console.error("TTS generation error:", err);
      return res.status(500).json({
        error: err?.message || "TTS generation failed",
        fallback: true,
      });
    }
  });

  // API 3: Gemini Vision Scene Analysis & Script Writing
  app.post("/api/analyze", async (req, res) => {
    try {
      const { imageBase64, mimeType = "image/jpeg" } = req.body;
      const ai = getGenAI();

      if (!ai) {
        return res.status(503).json({ error: "GEMINI_API_KEY not configured" });
      }

      const prompt = `あなたは世界的な紀行映画・ドキュメンタリー番組の監督・脚本家です。この写真の情景、光、色彩、雰囲気を深く読み取り、観る者の心に染み入る演出テキストを次のJSON形式で出力してください：
【最重要要件】：
スライドショーの心地よいテンポを保つため、ナレーション朗読台本（script）は【10秒前後で朗読できる長さ（日本語で42〜48文字程度、1〜2文）】を目安に作成してください。長すぎず、写真の余韻とBGMが心地よく響く簡潔で情感あふれる美しい日本語にしてください。
{
  "title": "詩的なロケーション/タイトル (例: 悠久の古都 - 静寂の小径)",
  "subtitle": "映画のワンシーンのような文学的字幕 (1文、15〜25文字程度)",
  "script": "落ち着いた語り部が10秒前後で穏やかに語るナレーション朗読台本 (厳密に42〜48文字程度、1〜2文)"
}`;

      const parts: any[] = [{ text: prompt }];
      if (imageBase64) {
        const cleanBase64 = imageBase64.includes(",")
          ? imageBase64.split(",")[1]
          : imageBase64;
        parts.push({
          inlineData: {
            mimeType,
            data: cleanBase64,
          },
        });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: [{ parts }],
        config: {
          responseMimeType: "application/json",
        },
      });

      const text = response.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!text) {
        throw new Error("No text returned from Gemini");
      }

      const parsed = JSON.parse(text);
      return res.json(parsed);
    } catch (err: any) {
      console.error("Gemini Analysis error:", err);
      return res.status(500).json({ error: err?.message || "Analysis failed" });
    }
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
