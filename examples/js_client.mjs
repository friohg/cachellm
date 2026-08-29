// JavaScript / TypeScript OpenAI SDK against CacheLLM.
//
//   npm install openai
//   node examples/js_client.mjs
//
// Only baseURL changes.

import OpenAI from "openai";

const PROXY = "http://localhost:4000/v1";
const MODEL = "mock-model"; // whatever your upstream serves

const client = new OpenAI({
  baseURL: PROXY,
  apiKey: "not-needed-when-the-proxy-holds-the-key", // SDK requires a value
});

const messages = [
  { role: "system", content: "You are a concise assistant." },
  { role: "user", content: "Explain what an LLM response cache does, in one sentence." },
];

async function timedCall(label) {
  const started = performance.now();
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0,
  });
  const ms = (performance.now() - started).toFixed(1);
  console.log(`${label.padEnd(12)} ${ms.padStart(8)} ms  ${completion.choices[0].message.content}`);
}

async function showCacheHeader() {
  // The raw fetch shows which layer answered.
  const response = await fetch(`${PROXY}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages, temperature: 0 }),
  });
  console.log("X-CacheLLM-Cache:", response.headers.get("x-cachellm-cache"));
  console.log("X-CacheLLM-Key:  ", response.headers.get("x-cachellm-key"));
  await response.text();
}

async function streamingDemo() {
  console.log("\nstreaming (cache replay keeps normal streaming behaviour):");
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0,
    stream: true,
  });
  for await (const chunk of stream) {
    const piece = chunk.choices?.[0]?.delta?.content;
    if (piece) process.stdout.write(piece);
  }
  process.stdout.write("\n");
}

async function stats() {
  const data = await (await fetch("http://localhost:4000/api/stats")).json();
  const c = data.counters;
  console.log(
    `\nrequests=${c.total_requests} hits=${c.cache_hits} misses=${c.cache_misses} ` +
      `hit_rate=${data.cache_hit_rate_pct}% saved=${data.cost.savings} ${data.cost.currency}`
  );
}

await timedCall("first call"); // CACHE MISS -> upstream
await timedCall("second call"); // CACHE HIT  -> local
await showCacheHeader();
await streamingDemo();
await stats();
