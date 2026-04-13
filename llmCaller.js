/**
 * Isolated LLM caller for assignment feature.
 * Uses same GROQ_API_KEY env var but NEVER touches host platform's _call_groq.
 * Token limits per section for efficiency.
 */
const https = require('https');

const GROQ_URL  = 'https://api.groq.com/openai/v1/chat/completions';
const MODEL     = 'llama-3.3-70b-versatile';
const FALLBACK  = 'llama-3.1-8b-instant';

function _post(payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) return reject(new Error('GROQ_API_KEY not set'));

    const options = {
      hostname: 'api.groq.com',
      path: '/openai/v1/chat/completions',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: 45000,
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (res.statusCode !== 200) return reject(new Error(`Groq ${res.statusCode}: ${data.slice(0,200)}`));
          resolve(parsed.choices[0].message.content.trim());
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('LLM timeout')); });
    req.write(body);
    req.end();
  });
}

/**
 * generateSection(prompt, maxTokens)
 * Isolated section-wise LLM call. Falls back to smaller model on 429.
 */
async function generateSection(prompt, maxTokens = 800) {
  const payload = (model) => ({
    model,
    messages: [
      { role: 'system', content: 'You are a legal academic writer. Write formally, precisely, in proper academic style.' },
      { role: 'user',   content: prompt },
    ],
    temperature: 0.3,
    max_tokens: maxTokens,
  });

  try {
    return await _post(payload(MODEL));
  } catch (err) {
    if (err.message.includes('429') || err.message.includes('503')) {
      await new Promise(r => setTimeout(r, 1200));
      return await _post(payload(FALLBACK));
    }
    throw err;
  }
}

module.exports = { generateSection };
