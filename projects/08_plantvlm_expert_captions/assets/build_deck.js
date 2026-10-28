/* PlantVLM — 12-slide introductory technical deck.
   Palette: dark green + white only. Fonts: Cambria (headings) / Calibri (body).
   Run:  node assets/build_deck.js
*/
const pptxgen = require("pptxgenjs");
const path = require("path");

const SHOTS = path.join(__dirname, "screenshots");
const DATA = path.join(__dirname, "..", "1stJuly", "Dataset");

// ── palette ────────────────────────────────────────────────────────────────
const INK = "14342A";   // deep green — dark slides, headings
const GREEN = "2E6B4F"; // mid green  — accents, numbers
const TINT = "EDF3EF";  // pale green — card fills
const LINE = "C9DAD0";  // hairline
const BODY = "1C2620";  // near-black body text
const MUTE = "5C6B62";  // muted label text
const WHITE = "FFFFFF";

const HEAD = "Cambria";
const TEXT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Harish · Aakash · Kamaksha";
pres.title = "Building a Vision-Language Model for Tomato Leaf Disease";

// ── helpers ────────────────────────────────────────────────────────────────
const card = (s, x, y, w, h, fill) =>
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || TINT }, line: { color: LINE, width: 0.75 },
  });

const darkCard = (s, x, y, w, h) =>
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: INK }, line: { type: "none" },
  });

const title = (s, t, sub) => {
  s.addText(t, {
    x: 0.6, y: 0.42, w: 12.1, h: 0.62, margin: 0,
    fontFace: HEAD, fontSize: 32, bold: true, color: INK, align: "left",
  });
  if (sub)
    s.addText(sub, {
      x: 0.6, y: 1.06, w: 12.1, h: 0.34, margin: 0,
      fontFace: TEXT, fontSize: 14, color: MUTE, align: "left",
    });
};

const chevron = (s, x, y) =>
  s.addShape(pres.ShapeType.rightTriangle, {
    x, y, w: 0.16, h: 0.22, fill: { color: GREEN }, line: { type: "none" }, rotate: 0,
  });

const arrowDown = (s, x, y, h) =>
  s.addShape(pres.ShapeType.line, {
    x, y, w: 0, h,
    line: { color: GREEN, width: 1.5, endArrowType: "triangle" },
  });

const arrowRight = (s, x, y, w) =>
  s.addShape(pres.ShapeType.line, {
    x, y, w, h: 0,
    line: { color: GREEN, width: 1.5, endArrowType: "triangle" },
  });

const numDot = (s, x, y, n) => {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.34, h: 0.34, fill: { color: GREEN }, line: { type: "none" },
  });
  s.addText(String(n), {
    x, y, w: 0.34, h: 0.34, margin: 0,
    fontFace: TEXT, fontSize: 12, bold: true, color: WHITE,
    align: "center", valign: "middle",
  });
};

// ══════════════════════════════════════════════════════════ 1 · TITLE
{
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("Teaching a Model to Describe a Sick Leaf", {
    x: 0.9, y: 2.15, w: 11.5, h: 1.0, margin: 0,
    fontFace: HEAD, fontSize: 40, bold: true, color: WHITE,
  });
  s.addText("How we built a Vision-Language Model pipeline for tomato leaf disease", {
    x: 0.9, y: 3.2, w: 11.5, h: 0.45, margin: 0,
    fontFace: TEXT, fontSize: 18, color: "A9C6B5",
  });

  s.addShape(pres.ShapeType.line, {
    x: 0.95, y: 3.95, w: 2.2, h: 0, line: { color: GREEN, width: 2 },
  });

  s.addText("Harish  ·  Aakash  ·  Kamaksha", {
    x: 0.9, y: 4.25, w: 11.5, h: 0.35, margin: 0,
    fontFace: TEXT, fontSize: 15, color: WHITE,
  });
  s.addText("Qwen2.5-VL-7B  ·  LoRA fine-tuning  ·  150 manually validated captions",
    { x: 0.9, y: 4.68, w: 11.5, h: 0.32, margin: 0,
      fontFace: TEXT, fontSize: 12, color: "8FB3A0" });

  s.addNotes("One sentence to open: a classifier tells you a disease name. We wanted a model that tells you what it actually sees on the leaf — and this is how we built it.");
}

// ══════════════════════════════════════════════════════════ 2 · WHY VLMs
{
  const s = pres.addSlide();
  title(s, "Three kinds of model", "What each one takes in, and what it can give back");

  const cols = [
    { t: "CNN", sub: "Classifier",
      inp: "Image", out: "One label",
      says: "Tomato_Early_blight",
      note: "No reason. No detail.\nYou cannot check it." },
    { t: "LLM", sub: "Language model",
      inp: "Text", out: "Text",
      says: "“Early blight is caused by\nAlternaria solani…”",
      note: "Fluent — but it has\nnever seen your leaf." },
    { t: "VLM", sub: "Vision-Language model",
      inp: "Image + Question", out: "Sentences",
      says: "“Two brown patches, one near\nthe upper centre…”",
      note: "An answer with the\nevidence attached." },
  ];

  cols.forEach((c, i) => {
    const x = 0.6 + i * 4.15;
    const highlight = i === 2;
    if (highlight) darkCard(s, x, 1.65, 3.85, 4.3);
    else card(s, x, 1.65, 3.85, 4.3);

    s.addText(c.t, {
      x: x + 0.3, y: 1.83, w: 3.2, h: 0.5, margin: 0,
      fontFace: HEAD, fontSize: 24, bold: true,
      color: highlight ? WHITE : INK,
    });
    s.addText(c.sub, {
      x: x + 0.3, y: 2.33, w: 3.2, h: 0.28, margin: 0,
      fontFace: TEXT, fontSize: 11, color: highlight ? "8FB3A0" : MUTE,
    });

    s.addText(`${c.inp}   →   ${c.out}`, {
      x: x + 0.3, y: 2.68, w: 3.3, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 12, bold: true,
      color: highlight ? "A9C6B5" : GREEN,
    });

    s.addShape(pres.ShapeType.line, {
      x: x + 0.3, y: 3.08, w: 3.25, h: 0,
      line: { color: highlight ? "3A5C4A" : LINE, width: 0.75 },
    });

    s.addText(c.says, {
      x: x + 0.3, y: 3.22, w: 3.3, h: 1.0, margin: 0,
      fontFace: TEXT, fontSize: 11, italic: true,
      color: highlight ? WHITE : BODY,
    });
    s.addText(c.note, {
      x: x + 0.3, y: 4.6, w: 3.3, h: 1.1, margin: 0,
      fontFace: TEXT, fontSize: 12, color: highlight ? "A9C6B5" : MUTE,
    });
  });

  s.addText("A farmer cannot act on a label. A description can be read, checked, and questioned.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.4, margin: 0,
      fontFace: TEXT, fontSize: 14, bold: true, color: INK });

  s.addNotes("Ask the room what they would build. They will say classifier. Then show what a label cannot do.");
}

// ══════════════════════════════════════════════════════════ 3 · WHY IT IS HARD
{
  const s = pres.addSlide();
  title(s, "Why training a VLM is hard", "Six problems we had to work around");

  const items = [
    ["Data hunger", "VLMs are pre-trained on millions of image-text pairs. We had thirty images."],
    ["Cost", "A 7B model's full weights do not fit in 24 GB of VRAM."],
    ["Alignment", "The model must connect what it sees to what it says — not just describe pixels."],
    ["Hallucination", "It will confidently describe symptoms that are not in the picture."],
    ["Domain knowledge", "General models have barely seen plant pathology."],
    ["No datasets", "Crop images come with labels, almost never with sentences."],
  ];

  items.forEach((it, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.15, y = 1.72 + row * 2.32;
    card(s, x, y, 3.85, 2.05);
    numDot(s, x + 0.3, y + 0.28, i + 1);
    s.addText(it[0], {
      x: x + 0.75, y: y + 0.3, w: 2.9, h: 0.32, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: INK,
    });
    s.addText(it[1], {
      x: x + 0.3, y: y + 0.82, w: 3.3, h: 1.0, margin: 0,
      fontFace: TEXT, fontSize: 12, color: BODY,
    });
  });

  s.addNotes("Keep this fast — one line each. The point is that none of these are solved by trying harder; each needs a design decision, which the rest of the talk shows.");
}

// ══════════════════════════════════════════════════════════ 4 · MODELS
{
  const s = pres.addSlide();
  title(s, "The models we put on the bench", "Open-weight VLMs we downloaded, ran, and fine-tuned on our data");

  const rows = [
    ["Qwen2.5-VL-7B-Instruct", "Alibaba", "Fine-tuned · strong, stable output", true],
    ["LLaVA-1.6-Mistral-7B", "LLaVA", "Fine-tuned · adapters trained", false],
    ["BLIP-2 OPT-2.7B", "Salesforce", "Fine-tuned · lost the output format", false],
    ["BLIP-2 FLAN-T5-XL", "Salesforce", "Evaluated", false],
    ["Florence-2-base-ft", "Microsoft", "Attempted · needed source patching", false],
  ];

  rows.forEach((r, i) => {
    const y = 1.75 + i * 0.86;
    if (r[3]) darkCard(s, 0.6, y, 12.1, 0.72);
    else card(s, 0.6, y, 12.1, 0.72, WHITE);
    s.addText(r[0], {
      x: 0.95, y: y + 0.06, w: 4.6, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 15, bold: true, color: r[3] ? WHITE : BODY,
    });
    s.addText(r[1], {
      x: 5.6, y: y + 0.06, w: 2.2, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12, color: r[3] ? "8FB3A0" : MUTE,
    });
    s.addText(r[2], {
      x: 7.9, y: y + 0.06, w: 4.5, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12, color: r[3] ? "A9C6B5" : MUTE,
    });
  });

  s.addText("No model is chosen from a leaderboard. We ran every one on our own leaves.",
    { x: 0.6, y: 6.32, w: 12.1, h: 0.4, margin: 0,
      fontFace: TEXT, fontSize: 14, bold: true, color: INK });

  s.addNotes("Say plainly: benchmark numbers are published on generic datasets. Plant leaves on a plain background are not that, so we tested on our data.");
}

// ══════════════════════════════════════════════════════════ 5 · BENCHMARKING
{
  const s = pres.addSlide();
  title(s, "How we compared them", "Same images, same prompt, side-by-side outputs");

  const steps = ["Our\ntomato\nimages", "One fixed\nprompt", "Run every\nVLM", "Collect\noutputs",
                 "Read them\nside by side", "Select the\nbest model"];
  steps.forEach((t, i) => {
    const x = 0.6 + i * 2.08;
    const last = i === steps.length - 1;
    if (last) darkCard(s, x, 2.15, 1.78, 1.65);
    else card(s, x, 2.15, 1.78, 1.65);
    s.addText(t, {
      x: x + 0.08, y: 2.25, w: 1.62, h: 1.45, margin: 0,
      fontFace: TEXT, fontSize: 12, bold: true, align: "center", valign: "middle",
      color: last ? WHITE : BODY,
    });
    if (i < steps.length - 1) arrowRight(s, x + 1.82, 2.98, 0.22);
  });

  card(s, 0.6, 4.3, 5.9, 1.75);
  s.addText("One-shot benchmarking", {
    x: 0.95, y: 4.5, w: 5.2, h: 0.34, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: INK,
  });
  s.addText("Every model sees each leaf once, with no examples and no hints — exactly how it would meet a new photograph.",
    { x: 0.95, y: 4.92, w: 5.25, h: 0.95, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: BODY });

  card(s, 6.8, 4.3, 5.9, 1.75);
  s.addText("Qualitative comparison", {
    x: 7.15, y: 4.5, w: 5.2, h: 0.34, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: INK,
  });
  s.addText("We read the outputs. Is it describing this leaf, or reciting textbook facts? Does it hold a usable format?",
    { x: 7.15, y: 4.92, w: 5.25, h: 0.95, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: BODY });

  s.addNotes("Emphasise: at this stage we were not chasing a metric. We were reading text and asking whether it was grounded in the image in front of it.");
}

// ══════════════════════════════════════════════════════════ 6 · WHY QWEN
{
  const s = pres.addSlide();
  title(s, "Why Qwen2.5-VL", "It described the leaf in front of it, and kept doing so");

  const reasons = [
    ["Visual grounding", "Describes marks that are actually on this leaf, with positions."],
    ["Agricultural sense", "Recognises lesions, chlorosis and leaf curling without coaching."],
    ["Consistency", "Same prompt, same shape of answer, image after image."],
    ["Stability", "Holds the output format through fine-tuning — several models did not."],
  ];
  reasons.forEach((r, i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.72 + Math.floor(i / 2) * 1.42;
    card(s, x, y, 5.9, 1.22);
    numDot(s, x + 0.32, y + 0.24, i + 1);
    s.addText(r[0], {
      x: x + 0.8, y: y + 0.24, w: 4.9, h: 0.32, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: INK,
    });
    s.addText(r[1], {
      x: x + 0.8, y: y + 0.62, w: 4.9, h: 0.5, margin: 0,
      fontFace: TEXT, fontSize: 12, color: BODY,
    });
  });

  darkCard(s, 0.6, 4.72, 12.1, 1.5);
  s.addText("What ruled a model out", {
    x: 0.95, y: 4.9, w: 5.0, h: 0.32, margin: 0,
    fontFace: HEAD, fontSize: 16, bold: true, color: WHITE,
  });
  s.addText("BLIP-2 produced text, but after fine-tuning it never kept our field structure — so not one of its 200 test outputs could be read back automatically. Qwen held the format every time.",
    { x: 0.95, y: 5.3, w: 11.4, h: 0.8, margin: 0,
      fontFace: TEXT, fontSize: 13, color: "A9C6B5" });

  s.addNotes("The BLIP-2 story is the useful one: it is not that the model was bad at English, it is that it would not hold a structure we could parse. Format compliance is a real capability difference.");
}

// ══════════════════════════════════════════════════════════ 7 · WORKFLOW (BIG)
{
  const s = pres.addSlide();
  title(s, "Our workflow", "From a folder of leaf photographs to a fine-tuned model");

  // Phase 1
  card(s, 0.6, 1.62, 5.75, 4.55);
  s.addText("PHASE 1  ·  PROVING THE PIPELINE", {
    x: 0.95, y: 1.8, w: 5.1, h: 0.28, margin: 0,
    fontFace: TEXT, fontSize: 11, bold: true, color: GREEN, charSpacing: 1,
  });

  const p1 = ["Tomato dataset", "Prompt engineering", "Captions built from labels",
              "LoRA fine-tuning", "Working demo app"];
  p1.forEach((t, i) => {
    const y = 2.24 + i * 0.72;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.95, y, w: 5.05, h: 0.52, rectRadius: 0.06,
      fill: { color: WHITE }, line: { color: LINE, width: 0.75 },
    });
    s.addText(t, {
      x: 1.15, y, w: 4.7, h: 0.52, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13, color: BODY,
    });
    if (i < p1.length - 1) arrowDown(s, 3.47, y + 0.53, 0.18);
  });

  // Phase 2
  darkCard(s, 6.95, 1.62, 5.75, 4.55);
  s.addText("PHASE 2  ·  REAL SUPERVISION", {
    x: 7.3, y: 1.8, w: 5.1, h: 0.28, margin: 0,
    fontFace: TEXT, fontSize: 11, bold: true, color: "8FB3A0", charSpacing: 1,
  });

  const p2 = ["Humans describe every leaf", "Mentor reviews every caption",
              "150 validated image-caption pairs", "Flashcard learning",
              "LoRA fine-tuning → inference → evaluation"];
  p2.forEach((t, i) => {
    const y = 2.24 + i * 0.72;
    s.addShape(pres.ShapeType.roundRect, {
      x: 7.3, y, w: 5.05, h: 0.52, rectRadius: 0.06,
      fill: { color: "1D4536" }, line: { type: "none" },
    });
    s.addText(t, {
      x: 7.5, y, w: 4.7, h: 0.52, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13, color: WHITE,
    });
    if (i < p2.length - 1) arrowDown(s, 9.82, y + 0.53, 0.18);
  });

  s.addText("The validated captions are now our only supervision data.", {
    x: 0.6, y: 6.35, w: 12.1, h: 0.35, margin: 0,
    fontFace: TEXT, fontSize: 14, bold: true, color: INK,
  });

  s.addNotes("This is the heart of the talk. Phase 1 proved the machinery works end to end. Phase 2 replaced the weakest part of it — where the sentences came from — with human work.");
}

// ══════════════════════════════════════════════════════════ 8 · FLASHCARDS
{
  const s = pres.addSlide();
  title(s, "Flashcard learning", "One leaf, five correct answers — a real example from our dataset");

  s.addImage({
    path: path.join(DATA, "early_blight", "4.JPG"),
    x: 0.6, y: 1.75, w: 3.3, h: 3.3,
  });
  s.addText("Tomato · Early Blight · image 4", {
    x: 0.6, y: 5.12, w: 3.3, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11, color: MUTE, align: "center",
  });

  const caps = [
    "Two prominent brown patches stand out against the dark green background…",
    "The leaf surface contains a mix of green tissue and several darker damaged regions…",
    "The discoloration is distributed across separate locations rather than one continuous area…",
    "Dark green leaf with multiple isolated lesions. Largest mark located on the right side…",
    "Several damaged sections interrupt an otherwise dark green leaf surface…",
  ];
  caps.forEach((c, i) => {
    const y = 1.75 + i * 0.72;
    card(s, 4.35, y, 8.35, 0.6, WHITE);
    s.addShape(pres.ShapeType.ellipse, {
      x: 4.55, y: y + 0.14, w: 0.32, h: 0.32,
      fill: { color: GREEN }, line: { type: "none" },
    });
    s.addText(String(i + 1), {
      x: 4.55, y: y + 0.14, w: 0.32, h: 0.32, margin: 0,
      fontFace: TEXT, fontSize: 11, bold: true, color: WHITE,
      align: "center", valign: "middle",
    });
    s.addText(c, {
      x: 5.02, y, w: 7.5, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 11.5, color: BODY,
    });
    arrowRight(s, 3.98, y + 0.3, 0.3);
  });

  darkCard(s, 0.6, 5.55, 12.1, 0.85);
  s.addText("Five different sentences, all true. The model cannot memorise one — so it learns the leaf.",
    { x: 0.95, y: 5.55, w: 11.4, h: 0.85, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 14, bold: true, color: WHITE });

  s.addNotes("Hold on this slide. The five captions are not duplicates — they are five valid descriptions. Because no single sentence minimises the error, the model is pushed toward the visual evidence they share.");
}

// ══════════════════════════════════════════════════════════ 9 · LoRA
{
  const s = pres.addSlide();
  title(s, "Fine-tuning without retraining", "We add to the model instead of rewriting it");

  darkCard(s, 0.6, 1.75, 6.2, 3.5);
  s.addText("Qwen2.5-VL  ·  frozen", {
    x: 0.95, y: 1.95, w: 5.5, h: 0.42, margin: 0,
    fontFace: HEAD, fontSize: 18, bold: true, color: WHITE,
  });
  s.addText("Everything it already knows — what a leaf is, what “brown” means, how to build a sentence. We change none of it.",
    { x: 0.95, y: 2.42, w: 5.5, h: 0.85, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: "A9C6B5" });

  for (let i = 0; i < 3; i++) {
    s.addShape(pres.ShapeType.roundRect, {
      x: 1.25 + i * 1.85, y: 3.55, w: 1.5, h: 1.15, rectRadius: 0.06,
      fill: { color: TINT }, line: { type: "none" },
    });
    s.addText("LoRA\nadapter", {
      x: 1.25 + i * 1.85, y: 3.55, w: 1.5, h: 1.15, margin: 0,
      fontFace: TEXT, fontSize: 12, bold: true, color: INK,
      align: "center", valign: "middle",
    });
  }
  s.addText("small new weights, slotted in", {
    x: 0.95, y: 4.82, w: 5.5, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11, italic: true, color: "8FB3A0", align: "center",
  });

  const stats = [
    ["0.484%", "of the model's weights are trained"],
    ["173 MB", "the entire saved result"],
    ["~20 min", "one full fine-tuning run"],
    ["19.4 / 24 GB", "peak memory on our workstation"],
  ];
  stats.forEach((st, i) => {
    const y = 1.75 + i * 0.9;
    card(s, 7.15, y, 5.55, 0.76, WHITE);
    s.addText(st[0], {
      x: 7.45, y, w: 2.3, h: 0.76, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 20, bold: true, color: GREEN,
    });
    s.addText(st[1], {
      x: 9.75, y, w: 2.75, h: 0.76, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 11.5, color: BODY,
    });
  });

  s.addText("The textbook stays untouched. We only add sticky notes in the right margins.", {
    x: 0.6, y: 5.62, w: 12.1, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 14, bold: true, color: INK,
  });
  s.addText("Peel the adapters off and the original model is exactly as it was — so one base model can serve many crops.",
    { x: 0.6, y: 6.02, w: 12.1, h: 0.4, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: MUTE });

  s.addNotes("Use the textbook analogy out loud. Full fine-tuning is rewriting the book; LoRA is sticky notes. Then give the four numbers — they make it concrete.");
}

// ══════════════════════════════════════════════════════════ 10 · DEMO
{
  const s = pres.addSlide();
  title(s, "The system, running", "Upload a leaf → the model reads it → a structured diagnosis comes back");

  s.addImage({ path: path.join(SHOTS, "shot_diagnosis.png"),
               x: 0.6, y: 1.72, w: 8.5, h: 3.31 });
  s.addImage({ path: path.join(SHOTS, "shot_metrics.png"),
               x: 9.35, y: 1.72, w: 1.55, h: 3.52 });
  s.addImage({ path: path.join(SHOTS, "shot_log.png"),
               x: 11.15, y: 1.72, w: 1.55, h: 0.86 });

  s.addText("Live app", {
    x: 9.35, y: 5.3, w: 1.55, h: 0.26, margin: 0,
    fontFace: TEXT, fontSize: 9.5, color: MUTE, align: "center",
  });
  s.addText("Every prediction logged", {
    x: 11.15, y: 2.64, w: 1.55, h: 0.5, margin: 0,
    fontFace: TEXT, fontSize: 9.5, color: MUTE, align: "center",
  });

  const flow = ["Upload leaf", "Not-a-leaf check", "Model reads image",
                "Structured diagnosis", "Logged to disk"];
  flow.forEach((t, i) => {
    const x = 0.6 + i * 2.48;
    card(s, x, 5.55, 2.2, 0.7, WHITE);
    s.addText(t, {
      x: x + 0.08, y: 5.55, w: 2.04, h: 0.7, margin: 0,
      fontFace: TEXT, fontSize: 11, bold: true, color: BODY,
      align: "center", valign: "middle",
    });
    if (i < flow.length - 1) arrowRight(s, x + 2.24, 5.9, 0.2);
  });

  s.addNotes("This is a real screenshot from the app running on our workstation — a tomato early blight leaf, correctly identified, with the confidence the model reports. If the GPU is free, run it live instead.");
}

// ══════════════════════════════════════════════════════════ 11 · RESULTS
{
  const s = pres.addSlide();
  title(s, "Before and after", "The same kind of leaf, described by the same model — untuned, then fine-tuned");

  card(s, 0.6, 1.7, 5.9, 4.05, WHITE);
  s.addText("BASE Qwen2.5-VL", {
    x: 0.95, y: 1.92, w: 5.2, h: 0.32, margin: 0,
    fontFace: TEXT, fontSize: 11, bold: true, color: MUTE, charSpacing: 1,
  });
  s.addText("“The tomato leaf in the image appears to have some signs of disease or damage. There are dark, irregularly shaped spots… which could indicate various issues such as fungal infections, bacterial diseases, or physical damage from pests or environmental stressors… it would be advisable to consult with a horticulturist or plant pathologist for a”",
    { x: 0.95, y: 2.35, w: 5.25, h: 2.2, margin: 0,
      fontFace: TEXT, fontSize: 11.5, italic: true, color: BODY });
  ["Hedges across three possibilities", "Gives advice nobody asked for",
   "Runs past the length limit mid-sentence"].forEach((t, i) => {
    s.addText("—  " + t, {
      x: 0.95, y: 4.72 + i * 0.32, w: 5.25, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 11, color: MUTE });
  });

  darkCard(s, 6.8, 1.7, 5.9, 4.05);
  s.addText("AFTER FINE-TUNING", {
    x: 7.15, y: 1.92, w: 5.2, h: 0.32, margin: 0,
    fontFace: TEXT, fontSize: 11, bold: true, color: "8FB3A0", charSpacing: 1,
  });
  s.addText("“Plant: Tomato. Condition: Target Spot. Severity: Moderate. Circular brown spots with concentric lighter zones and chlorotic halos are scattered on this leaf surface.”",
    { x: 7.15, y: 2.35, w: 5.25, h: 1.5, margin: 0,
      fontFace: TEXT, fontSize: 12.5, italic: true, color: WHITE });
  ["Names what it sees", "Same shape every time",
   "Short enough to read at a glance"].forEach((t, i) => {
    s.addText("—  " + t, {
      x: 7.15, y: 4.72 + i * 0.32, w: 5.25, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 11, color: "A9C6B5" });
  });

  s.addText("Both outputs are real, from our own runs. Same model — the difference is what we taught it to say.",
    { x: 0.6, y: 6.0, w: 12.1, h: 0.4, margin: 0,
      fontFace: TEXT, fontSize: 13.5, bold: true, color: INK });

  s.addNotes("Read the base output aloud — the rambling is the point. Then read the tuned one. Nothing about the model changed except the sentences we showed it.");
}

// ══════════════════════════════════════════════════════════ 12 · CLOSING
{
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("Where we are", {
    x: 0.9, y: 0.75, w: 5.4, h: 0.5, margin: 0,
    fontFace: HEAD, fontSize: 26, bold: true, color: WHITE,
  });

  const now = [
    "6 tomato classes  ·  30 images  ·  150 validated captions",
    "Pipeline built, verified, and reproducible end to end",
    "Fine-tuning ready to run on our own workstation",
  ];
  now.forEach((t, i) => {
    s.addText("—  " + t, {
      x: 0.9, y: 1.42 + i * 0.42, w: 5.6, h: 0.38, margin: 0,
      fontFace: TEXT, fontSize: 13, color: "A9C6B5" });
  });

  s.addText("What comes next", {
    x: 7.0, y: 0.75, w: 5.4, h: 0.5, margin: 0,
    fontFace: HEAD, fontSize: 26, bold: true, color: WHITE,
  });
  const next = ["More validated images", "More crops beyond tomato",
                "Newer and larger VLMs", "A wider benchmark",
                "Disease diagnosis, not only description", "Explaining why the model said it"];
  next.forEach((t, i) => {
    s.addText("—  " + t, {
      x: 7.0, y: 1.42 + i * 0.42, w: 5.4, h: 0.38, margin: 0,
      fontFace: TEXT, fontSize: 13, color: "A9C6B5" });
  });

  s.addShape(pres.ShapeType.line, {
    x: 0.95, y: 4.35, w: 11.4, h: 0, line: { color: "3A5C4A", width: 1 },
  });

  s.addText("Thank you", {
    x: 0.9, y: 4.75, w: 11.5, h: 0.75, margin: 0,
    fontFace: HEAD, fontSize: 38, bold: true, color: WHITE,
  });
  s.addText("Questions welcome — including the ones we have not answered yet.", {
    x: 0.9, y: 5.55, w: 11.5, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 15, color: "8FB3A0",
  });
  s.addText("Harish  ·  Aakash  ·  Kamaksha", {
    x: 0.9, y: 6.35, w: 11.5, h: 0.35, margin: 0,
    fontFace: TEXT, fontSize: 12, color: "5F8570",
  });

  s.addNotes("Close on the honest note: thirty images is a start, not a result. The methodology is what we are presenting.");
}

const out = path.join(__dirname, "..", "PlantVLM_Presentation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("written:", out));
