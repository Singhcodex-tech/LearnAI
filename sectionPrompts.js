/**
 * Prompt templates for each assignment section.
 * Ultra-short by design — max token efficiency.
 * Each fn returns a string prompt ready for LLM.
 */

const BASE = (section, topic) =>
  `Write ${section} for law assignment on: "${topic}". Formal, concise, structured. Include legal reasoning. Use [CASE: description] placeholders for unknown case laws.`;

const PROMPTS = {
  introduction: (topic) =>
    `${BASE('an Introduction (250-300 words)', topic)} Cover: definition, scope, legal significance, research questions.`,

  body: (topic) =>
    `${BASE('a Body / Legal Analysis (600-800 words)', topic)} Include: 3-4 sub-headings, statutory provisions, judicial interpretation, [CASE: relevant facts] placeholders, critical analysis.`,

  conclusion: (topic) =>
    `${BASE('a Conclusion (150-200 words)', topic)} Summarise findings, legal implications, recommendations.`,

  bibliography: (topic) =>
    `List 6-8 bibliography entries in Bluebook format for a law assignment on: "${topic}". Include mix of: cases, statutes, books, journal articles. Use [CITATION NEEDED] for uncertain details.`,

  declaration: (meta) =>
    `Write a 1-paragraph student declaration for a law assignment. Student: ${meta.student_name}, Roll No: ${meta.roll_no}, Course: ${meta.course}, College: ${meta.college}. Standard academic honesty declaration. Formal tone.`,

  acknowledgment: (meta) =>
    `Write a brief (100-word) acknowledgment for a law assignment by ${meta.student_name} from ${meta.college}. Thank faculty guide, institution library, and family. Formal academic tone.`,
};

module.exports = PROMPTS;
