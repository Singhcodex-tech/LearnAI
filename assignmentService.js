/**
 * assignmentService.js
 * Orchestrates section-wise generation. Lazy + cached.
 * No dependencies on host platform code.
 */
const { v4: uuidv4 } = require('uuid');
const store   = require('../utils/assignmentStore');
const llm     = require('../utils/llmCaller');
const PROMPTS = require('../prompts/sectionPrompts');

// Token budget per section (kept small for speed + cost)
const TOKEN_LIMITS = {
  declaration:     200,
  acknowledgment:  180,
  introduction:    450,
  body:            900,
  conclusion:      300,
  bibliography:    400,
};

/**
 * generateSection(sectionKey, topic, meta)
 * Lazy: only called when needed. Returns plain text.
 */
async function generateSection(sectionKey, topic, meta) {
  const promptFn = PROMPTS[sectionKey];
  if (!promptFn) throw new Error(`Unknown section: ${sectionKey}`);
  const prompt = (sectionKey === 'declaration' || sectionKey === 'acknowledgment')
    ? promptFn(meta)
    : promptFn(topic);
  return llm.generateSection(prompt, TOKEN_LIMITS[sectionKey] || 500);
}

/**
 * assembleAssignment(meta)
 * Checks cache first (same topic = reuse). Generates all sections sequentially.
 * Returns assignment object saved to store.
 */
async function assembleAssignment(meta) {
  // Cache hit on same topic
  const cached = store.findByTopic(meta.topic);
  if (cached) return cached;

  const id = uuidv4();
  const sections = {};

  const sectionKeys = [
    'declaration',
    ...(meta.include_acknowledgment ? ['acknowledgment'] : []),
    'introduction',
    'body',
    'conclusion',
    'bibliography',
  ];

  for (const key of sectionKeys) {
    sections[key] = await generateSection(key, meta.topic, meta);
  }

  const assignment = { id, meta, sections };
  store.set(id, assignment);
  return assignment;
}

module.exports = { assembleAssignment, generateSection };
