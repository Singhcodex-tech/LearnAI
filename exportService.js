/**
 * exportService.js
 * Builds DOCX (docx npm) and PDF (LibreOffice headless).
 * Fully isolated — no shared platform export logic.
 */
const fs        = require('fs');
const path      = require('path');
const os        = require('os');
const { execSync } = require('child_process');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  AlignmentType, PageBreak, LevelFormat, TableOfContents,
  BorderStyle,
} = require('docx');
const store = require('../utils/assignmentStore');

// ─── Helpers ────────────────────────────────────────────────────────────────

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 28, font: 'Arial' })],
    spacing: { before: 360, after: 160 },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 24, font: 'Arial' })],
    spacing: { before: 240, after: 120 },
  });
}

function body(text) {
  if (!text) return [];
  return text.split('\n').filter(l => l.trim()).map(line =>
    new Paragraph({
      children: [new TextRun({ text: line.trim(), size: 24, font: 'Arial' })],
      spacing: { after: 160 },
      alignment: AlignmentType.JUSTIFIED,
    })
  );
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function ruled() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2E4057', space: 1 } },
    spacing: { after: 160 },
    children: [],
  });
}

function centred(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: opts.after || 200 },
    children: [new TextRun({ text, ...opts })],
  });
}

// ─── Cover Page ─────────────────────────────────────────────────────────────

function buildCoverPage(meta) {
  return [
    new Paragraph({ children: [], spacing: { after: 1200 } }), // top spacer
    centred(meta.college.toUpperCase(), { bold: true, size: 32, font: 'Arial', after: 240 }),
    centred(meta.course, { size: 26, font: 'Arial', after: 480 }),
    ruled(),
    centred('LAW ASSIGNMENT', { bold: true, size: 36, font: 'Arial', allCaps: true, after: 200 }),
    centred(`ON`, { size: 24, font: 'Arial', after: 200 }),
    centred(meta.topic.toUpperCase(), { bold: true, size: 28, font: 'Arial', color: '1A3A5C', after: 480 }),
    ruled(),
    new Paragraph({ children: [], spacing: { after: 800 } }),
    centred(`Submitted by`, { size: 22, font: 'Arial', after: 120 }),
    centred(meta.student_name, { bold: true, size: 26, font: 'Arial', after: 120 }),
    centred(`Roll No: ${meta.roll_no}`, { size: 22, font: 'Arial', after: 120 }),
    centred(new Date().toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' }), { size: 22, font: 'Arial', after: 0 }),
    pageBreak(),
  ];
}

// ─── TOC ────────────────────────────────────────────────────────────────────

function buildToc() {
  return [
    h1('Table of Contents'),
    new TableOfContents('Table of Contents', {
      hyperlink: true,
      headingStyleRange: '1-2',
    }),
    pageBreak(),
  ];
}

// ─── DOCX builder ───────────────────────────────────────────────────────────

function buildDocx(assignment) {
  const { meta, sections } = assignment;
  const children = [];

  // Cover
  children.push(...buildCoverPage(meta));

  // Declaration
  children.push(h1('Declaration'), ...body(sections.declaration), pageBreak());

  // Acknowledgment (optional)
  if (sections.acknowledgment) {
    children.push(h1('Acknowledgment'), ...body(sections.acknowledgment), pageBreak());
  }

  // TOC
  children.push(...buildToc());

  // Introduction
  children.push(h1('1. Introduction'), ...body(sections.introduction), pageBreak());

  // Body — split on sub-headings (lines ending with ':' or all-caps lines become h2)
  children.push(h1('2. Legal Analysis'));
  const bodyLines = (sections.body || '').split('\n').filter(l => l.trim());
  for (const line of bodyLines) {
    const trimmed = line.trim();
    if (/^#{1,2}\s/.test(trimmed)) {
      children.push(h2(trimmed.replace(/^#{1,2}\s/, '')));
    } else if (/^[A-Z][A-Z\s]+:$/.test(trimmed) || /^\d+\.\d+\s/.test(trimmed)) {
      children.push(h2(trimmed));
    } else {
      children.push(new Paragraph({
        children: [new TextRun({ text: trimmed, size: 24, font: 'Arial' })],
        spacing: { after: 160 },
        alignment: AlignmentType.JUSTIFIED,
      }));
    }
  }
  children.push(pageBreak());

  // Conclusion
  children.push(h1('3. Conclusion'), ...body(sections.conclusion), pageBreak());

  // Bibliography
  children.push(h1('4. Bibliography (Bluebook)'), ...body(sections.bibliography));

  return new Document({
    styles: {
      default: { document: { run: { font: 'Arial', size: 24 } } },
      paragraphStyles: [
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { size: 32, bold: true, font: 'Arial', color: '1A3A5C' },
          paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { size: 26, bold: true, font: 'Arial', color: '2E5F8A' },
          paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      ],
    },
    numbering: { config: [] },
    sections: [{
      properties: {
        page: {
          size:   { width: 11906, height: 16838 },        // A4
          margin: { top: 1440, right: 1260, bottom: 1440, left: 1800 }, // left=1.25in for binding
        },
      },
      children,
    }],
  });
}

// ─── Public API ─────────────────────────────────────────────────────────────

async function exportDocx(id) {
  const assignment = store.get(id);
  if (!assignment) throw new Error(`Assignment ${id} not found`);
  const doc = buildDocx(assignment);
  return Packer.toBuffer(doc);
}

async function exportPdf(id) {
  const assignment = store.get(id);
  if (!assignment) throw new Error(`Assignment ${id} not found`);

  const tmpDir  = os.tmpdir();
  const docxPath = path.join(tmpDir, `assignment_${id}.docx`);
  const pdfPath  = path.join(tmpDir, `assignment_${id}.pdf`);

  // Write DOCX to temp
  const buf = await exportDocx(id);
  fs.writeFileSync(docxPath, buf);

  // Convert via LibreOffice headless (available in most server environments)
  try {
    execSync(`libreoffice --headless --convert-to pdf --outdir "${tmpDir}" "${docxPath}"`, { timeout: 30000 });
  } catch {
    // Fallback: try soffice
    execSync(`soffice --headless --convert-to pdf --outdir "${tmpDir}" "${docxPath}"`, { timeout: 30000 });
  }

  if (!fs.existsSync(pdfPath)) throw new Error('PDF conversion failed — LibreOffice not available');
  const pdfBuf = fs.readFileSync(pdfPath);

  // Cleanup
  try { fs.unlinkSync(docxPath); fs.unlinkSync(pdfPath); } catch (_) {}
  return pdfBuf;
}

module.exports = { exportDocx, exportPdf };
