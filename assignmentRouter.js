const express = require('express');
const router  = express.Router();
const assignmentService = require('../services/assignmentService');
const exportService     = require('../services/exportService');

// POST /api/assignment/generate
// Body: { topic, student_name, roll_no, course, college, include_acknowledgment? }
router.post('/generate', async (req, res) => {
  const { topic, student_name, roll_no, course, college, include_acknowledgment = true } = req.body;
  if (!topic || !student_name || !roll_no || !course || !college)
    return res.status(400).json({ error: 'Missing required fields: topic, student_name, roll_no, course, college' });

  try {
    const meta = { topic, student_name, roll_no, course, college, include_acknowledgment };
    const assignment = await assignmentService.assembleAssignment(meta);
    res.json({ success: true, assignment_id: assignment.id, sections: Object.keys(assignment.sections) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/assignment/:id/docx
router.get('/:id/docx', async (req, res) => {
  try {
    const buf = await exportService.exportDocx(req.params.id);
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    res.setHeader('Content-Disposition', `attachment; filename="assignment_${req.params.id}.docx"`);
    res.send(buf);
  } catch (err) {
    res.status(404).json({ error: err.message });
  }
});

// GET /api/assignment/:id/pdf
router.get('/:id/pdf', async (req, res) => {
  try {
    const buf = await exportService.exportPdf(req.params.id);
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="assignment_${req.params.id}.pdf"`);
    res.send(buf);
  } catch (err) {
    res.status(404).json({ error: err.message });
  }
});

// GET /api/assignment/:id/status  — check cached state
router.get('/:id/status', (req, res) => {
  const store = require('../utils/assignmentStore');
  const a = store.get(req.params.id);
  if (!a) return res.status(404).json({ error: 'Not found' });
  res.json({ id: a.id, topic: a.meta.topic, sections: Object.keys(a.sections), created_at: a.created_at });
});

module.exports = router;
