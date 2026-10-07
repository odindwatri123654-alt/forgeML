const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, AlignmentType, TableOfContents,
  PageBreak, Footer, PageNumber,
} = require('docx');

const MONO = 'Consolas';
const BODY = 'Calibri';

// `код` внутри текста -> моноширинный шрифт, **жирный** -> bold
function runs(text, base = {}) {
  const out = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('`')) {
      out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 20, color: '9C2B5B', ...base }));
    } else {
      out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const p = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120 }, ...opts });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(t) });
const label = (t) => new Paragraph({ children: [new TextRun({ text: t, bold: true, color: '1F4E79' })], spacing: { before: 120, after: 60 } });
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, children: runs(t), spacing: { after: 60 } }));

function code(lines) {
  const arr = Array.isArray(lines) ? lines : lines.split('\n');
  return arr.map((line, i) => new Paragraph({
    children: [new TextRun({ text: line.length ? line : ' ', font: MONO, size: 18 })],
    shading: { type: ShadingType.CLEAR, fill: 'F3F4F6', color: 'auto' },
    spacing: { before: i === 0 ? 80 : 0, after: i === arr.length - 1 ? 120 : 0, line: 260 },
    indent: { left: 200, right: 200 },
  }));
}

// Карточка одного метода / функции / класса
function entry(e) {
  const out = [h3(e.name)];
  if (e.file) out.push(new Paragraph({ children: [new TextRun({ text: 'Файл: ', italics: true, color: '666666' }), new TextRun({ text: e.file, font: MONO, size: 18, color: '666666' })], spacing: { after: 60 } }));
  if (e.sig) out.push(...code(e.sig));
  if (e.what) { out.push(label('Что делает')); out.push(p(e.what)); }
  if (e.how) { out.push(label('Как работает')); out.push(...(Array.isArray(e.how) ? bullets(e.how) : [p(e.how)])); }
  if (e.example) { out.push(label('Пример')); out.push(...code(e.example)); }
  if (e.notes) { out.push(label('Важно')); out.push(...bullets(e.notes)); }
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: 'DCE6F1', color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: head ? [new TextRun({ text, bold: true })] : runs(text) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
           ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}

// Содержимое: секции из файлов day*.js
const ctx = { p, h1, h2, h3, label, bullets, code, entry, table };
const days = fs.readdirSync(__dirname).filter((f) => /^day\d+\.js$/.test(f)).sort()
  .map((f) => require('./' + f)(ctx));

const intro = require('./intro.js')(ctx, days.length);

const doc = new Document({
  creator: 'ForgeML',
  title: 'ForgeML — справочник по коду',
  styles: {
    default: { document: { run: { font: BODY, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 36, bold: true, color: '1F4E79', font: BODY }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, color: '2E75B6', font: BODY }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, color: '000000', font: BODY }, paragraph: { spacing: { before: 260, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ['ForgeML — справочник · стр. ', PageNumber.CURRENT], size: 18, color: '888888' })] })] }) },
    children: [...intro, ...days.flat()],
  }],
});

const out = process.argv[2] || 'ForgeML_reference.docx';
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log('written', out); });
