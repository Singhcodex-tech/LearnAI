/**
 * ─── HOST PLATFORM INTEGRATION ──────────────────────────────────────────────
 * Add ONLY these 2 lines to your existing Express app. Nothing else changes.
 *
 *   const assignmentFeature = require('./features/assignment');
 *   app.use('/api/assignment', assignmentFeature);
 *
 * To disable: comment both lines. Zero side-effects.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * ─── ENVIRONMENT ─────────────────────────────────────────────────────────────
 * Uses existing GROQ_API_KEY — no new env vars needed.
 *
 * ─── ADDITIONAL DEPS (run once from project root) ────────────────────────────
 *   npm install docx uuid
 *
 * ─── API REFERENCE ────────────────────────────────────────────────────────────
 *
 * POST /api/assignment/generate
 * {
 *   "topic": "Right to Privacy under Indian Constitution",
 *   "student_name": "Priya Sharma",
 *   "roll_no": "LAW2024-047",
 *   "course": "B.A. LL.B. (Hons.) — Semester V",
 *   "college": "National Law School of India University",
 *   "include_acknowledgment": true
 * }
 * → { success: true, assignment_id: "uuid", sections: ["declaration", ...] }
 *
 * GET /api/assignment/:id/docx  → downloads .docx
 * GET /api/assignment/:id/pdf   → downloads .pdf
 * GET /api/assignment/:id/status → { id, topic, sections[], created_at }
 *
 * ─── DEMO OUTPUT PREVIEW ──────────────────────────────────────────────────────
 *
 * COVER PAGE
 * ──────────
 * NATIONAL LAW SCHOOL OF INDIA UNIVERSITY
 * B.A. LL.B. (Hons.) — Semester V
 *
 * LAW ASSIGNMENT ON
 * RIGHT TO PRIVACY UNDER INDIAN CONSTITUTION
 *
 * Submitted by: Priya Sharma | Roll No: LAW2024-047 | 13 April 2025
 *
 * DECLARATION
 * ───────────
 * I, Priya Sharma (Roll No. LAW2024-047), hereby declare that this assignment
 * submitted for B.A. LL.B. (Hons.) at National Law School of India University
 * is my original work and has not been submitted elsewhere...
 *
 * INTRODUCTION
 * ────────────
 * The right to privacy, though not explicitly enumerated in the Constitution of
 * India, has been recognised as a fundamental right under Article 21 by the
 * Supreme Court in Justice K.S. Puttaswamy (Retd.) v. Union of India (2017) 10 SCC 1.
 * This assignment examines the constitutional basis, judicial evolution...
 *
 * BODY — LEGAL ANALYSIS
 * ──────────────────────
 * 2.1 Constitutional Basis
 * Privacy finds its roots in Articles 14, 19, and 21...
 * [CASE: Govind v. State of Madhya Pradesh — right to privacy as emanating from Art. 21]
 *
 * BIBLIOGRAPHY (BLUEBOOK)
 * ────────────────────────
 * 1. Justice K.S. Puttaswamy v. Union of India, (2017) 10 SCC 1.
 * 2. Kharak Singh v. State of U.P., AIR 1963 SC 1295.
 * 3. Menon, N.R. Madhava, Constitutional Law of India (7th ed., EBC 2022).
 */
