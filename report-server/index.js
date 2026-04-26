/**
 * report-server — thin Node/Express service that sits alongside the
 * Python pipeline and handles two browser-facing concerns:
 *
 *   POST /api/generate   Streams a Claude (sonnet-4-6) narrative back to
 *                        the frontend as plain text chunks.
 *   POST /api/export     Renders the report HTML in headless Chrome and
 *                        returns a binary PDF. Chart images are passed
 *                        in by the client (already captured via
 *                        html2canvas) so we do NOT need to re-render
 *                        Recharts on the server.
 *
 * The Python pipeline is the source of truth for mlData; this server
 * never touches the dataset itself.
 */

import "dotenv/config";
import express from "express";
import cors from "cors";
import Anthropic from "@anthropic-ai/sdk";
import puppeteer from "puppeteer";

const PORT = process.env.PORT || 4000;
const CLAUDE_MODEL = "claude-sonnet-4-6";
const BRAND_ORANGE = "#F97316";
const BRAND_ORANGE_LIGHT = "#FFF7ED";

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ─── /api/generate ─────────────────────────────────────────────────
function buildNarrativePrompt({ mlData, target_column, problem_type, customInstructions, allowedChartIds }) {
  const ids = Array.isArray(allowedChartIds) && allowedChartIds.length > 0
    ? allowedChartIds.map(Number).filter((n) => n >= 1 && n <= 6)
    : [1, 2, 3, 4, 5, 6];
  const idList = ids.join(", ");

  // Tell Claude what each allowed chart actually shows so the narrative
  // around each marker is concrete instead of generic filler.
  // mlData.chartData is now a Claude-generated array of chart configs
  // (see ChartRegistry.js), so look up each id and use its title + insight.
  const chartArray = Array.isArray(mlData?.chartData) ? mlData.chartData : [];
  const chartById = Object.fromEntries(
    chartArray.filter((c) => c && Number.isInteger(c.id)).map((c) => [c.id, c]),
  );
  const chartHints = ids.map((n) => {
    const cfg = chartById[n];
    if (!cfg) return `- Chart ${n} — (no config).`;
    const kind = cfg.chartType || "Chart";
    const tail = cfg.insight ? ` — ${cfg.insight}` : "";
    return `- Chart ${n} (${kind}) — ${cfg.title || `chart ${n}`}${tail}`;
  }).join("\n");

  // Structured developer-context data — Claude turns these into intro paragraphs + markdown pipe tables.
  const leaderboard = mlData?.fullLeaderboard || mlData?.leaderboard || [];
  const featureImportance = mlData?.featureImportance ?? [];
  const runtime = mlData?.runtime ?? [];
  const runtimeTotal = runtime.reduce((a, b) => a + (b?.seconds ?? 0), 0);
  const bestModel = leaderboard[0];
  const primaryMetric = bestModel?.metric ?? "score";
  const hasStd = leaderboard.some((r) => typeof r?.scoreStd === "number" && r.scoreStd > 0);

  const leaderboardLines = leaderboard
    .map((r) => {
      const base = `  #${r.rank} ${r.model}: score=${Number(r.score).toFixed(4)}`;
      const std = typeof r.scoreStd === "number" ? ` ± ${Number(r.scoreStd).toFixed(4)}` : "";
      return `${base}${std}`;
    })
    .join("\n");

  const topImportance = featureImportance[0]?.importance ?? 1;
  const featureImportanceLines = featureImportance
    .map((f, i) => {
      const rel = topImportance ? ((f.importance / topImportance) * 100).toFixed(1) : "0.0";
      const dir = f.direction === "positive"
        ? `Higher values → Higher '${target_column}'`
        : f.direction === "negative"
        ? `Higher values → Lower '${target_column}'`
        : "Mixed relationship";
      return `  #${i + 1} ${f.feature}: ${Number(f.importance).toFixed(6)} (${rel}% relative) — ${dir}`;
    })
    .join("\n");

  const runtimeLines = runtime
    .map((r, i) => {
      const pct = runtimeTotal ? ((r.seconds / runtimeTotal) * 100).toFixed(1) : "0.0";
      return `  Stage ${i + 1} — ${r.stage}: ${Number(r.seconds).toFixed(1)}s (${pct}% of total)`;
    })
    .join("\n");

  // Heuristic: how the winning model exposes feature importance. Used for the extraction-method sentence.
  const winningModel = (bestModel?.model || "").toLowerCase();
  const importanceMethod = /logistic|linear/.test(winningModel)
    ? "linear coefficients (|coef_|) from the winning " + bestModel?.model
    : /forest|gradient|boost|extra.trees|xgb|lgbm|catboost/.test(winningModel)
    ? "impurity-based feature_importances_ from the winning " + bestModel?.model
    : /tree/.test(winningModel)
    ? "impurity-based feature_importances_ from the winning " + bestModel?.model
    : "permutation importance from the winning " + (bestModel?.model || "model");

  return `You are a senior data analyst writing a comprehensive analytical report.
Your task is to generate a well-structured, professional .PDF report based on the ML analysis results, dataset overview, and chart visualizations provided below.

TARGET VARIABLE: '${target_column}' (${problem_type})
Dataset: ${mlData?.datasetMeta?.rows ?? "?"} rows, ${mlData?.datasetMeta?.cols ?? "?"} columns.
Best model: ${bestModel?.model ?? "n/a"} (${primaryMetric} = ${bestModel ? Number(bestModel.score).toFixed(4) : ""}).

Top features:
${featureImportance
  .slice(0, 8)
  .map((f) => `- ${f.feature} (importance ${Number(f.importance).toFixed(3)}, ${f.direction})`)
  .join("\n")}

AUTOML TOURNAMENT DATA:
  Models tested: ${leaderboard.length}
  Primary metric: ${primaryMetric}
  Winner: ${bestModel?.model ?? "n/a"}
  Standard deviation available: ${hasStd ? "YES — include a Standard Deviation column" : "NO — omit the Standard Deviation column"}
Full leaderboard (rank, model, score${hasStd ? ", std" : ""}):
${leaderboardLines || "  (none)"}

FEATURE IMPORTANCE DATA (extraction method: ${importanceMethod}):
${featureImportanceLines || "  (none)"}

STAGE RUNTIME DATA:
${runtimeLines || "  (none)"}
  Total pipeline runtime: ${runtimeTotal.toFixed(1)}s

${customInstructions ? `Custom instructions from user:\n${customInstructions}\n` : ""}

## 1. Executive Summary

Write a concise overview covering:
- What the dataset represents and its scope (rows, columns, domain)
- The primary objective of the analysis (predicting '${target_column}')
- A high-level summary of the most important findings (top 3 drivers and their impact), This should be presented in bullet points
- The best-performing model and its accuracy
- Make any key fact or figures bold in text

## 2. Key Findings

Write one analytical paragraph per major insight. Each paragraph MUST:
- Start with a bold subheading — add numbering like "2.1". to segregate the sections.
- under each chart, add a one line concise caption that simply states what the graph is (NO ADDITIONAL DETAILS). It should be italic and the color of the captions text should be grey.
- State the finding, explain the directional impact on '${target_column}', cite specific numbers from the statistical analysis, and describe what the referenced chart reveals.
- Reference the relevant chart naturally (e.g., "As illustrated in Chart 1..." or "Chart 3 reveals that...").
- On the line IMMEDIATELY after the paragraph that discusses a chart, emit the marker <chart id="N" /> on its own line.

Use every allowed chart exactly once. Tell an analytical story where each insight builds on the previous one. Charts must feel like natural visual evidence supporting the narrative.

## 3. Developer Section

This section is for technical users. It contains exactly three subsections, each using a "### " heading, in this order. Each subsection contains: (a) a 1-2 paragraph intro describing the method/setup, (b) a BOLD label line ending with a colon, (c) a markdown pipe table built strictly from the data provided above — no invented rows, no invented columns, (d) a short closing commentary paragraph interpreting the table.

### AutoML Tournament Results

Intro (1 paragraph): state how many models were tested, the cross-validation setup if mentioned, the primary metric used ('${primaryMetric}'), and that the tournament identified the winner. Mention the winner name and its score.

**Model Performance Leaderboard:**

${hasStd
  ? `| Rank | Algorithm | ${primaryMetric} Score | Standard Deviation |
|------|-----------|-----------------------|--------------------|
| 1 | ... | 0.XXXX | ±0.XXXX |`
  : `| Rank | Algorithm | ${primaryMetric} Score |
|------|-----------|-----------------------|
| 1 | ... | 0.XXXX |`}

One row per model in rank order from the data above. Then a closing paragraph commenting on the winner, margin over runners-up, and stability of scores.

### Feature Importance Rankings

Intro (1 paragraph): describe the extraction method (${importanceMethod}) and what it means. Comment on the dominant predictor and any notable gap between the top feature and the rest.

**Feature Importance Detailed Rankings:**

| Rank | Feature | Coefficient | Relative Importance | Directional Impact |
|------|---------|-------------|---------------------|--------------------|
| 1 | ... | X.XXXXXX | 100.0% | Higher values → Higher '${target_column}' |

One row per feature in rank order from the data above — use the raw importance number in the Coefficient column, the relative-percent value in the Relative Importance column, and the directional-impact string from the data. Then a closing paragraph interpreting what the importance distribution says about the drivers.

### Stage-by-Stage Runtime Breakdown

Intro (1 paragraph): state the total pipeline time (${runtimeTotal.toFixed(1)}s) and highlight which stage(s) dominated.

**Detailed Runtime Analysis:**

| Stage | Process | Duration (seconds) | Percentage of Total |
|-------|---------|--------------------|---------------------|
| 1 | ... | 0.X | X.X% |
| Total | Complete Pipeline | ${runtimeTotal.toFixed(1)} | 100.0% |

One row per stage in order from the data above, followed by the Total row. Then a closing paragraph interpreting the runtime profile.

WRITING GUIDELINES:
- Be analytical and insightful, not just descriptive.
- Use professional but accessible language.
- When referencing charts, describe what the visual reveals — don't just say "see Chart X".
- Highlight contrasts, gaps, and surprising findings.
- Use specific numbers to support claims.
- Keep the executive summary concise; expand detail in key findings.
- Do NOT emit any section numbered with decimals (no "2.1", "2.5", "3.2"). Top-level sections use "## 1.", "## 2.", "## 3." ONLY. Subsections of the Developer Section use "### " with plain text (no numbers).
- The page break before the Developer Section is handled by the exporter — just write "## 3. Developer Section" normally.
- Do NOT place any <chart /> markers in the Developer Section.

CHART EMBEDDING RULES:
Reference ONLY these chart ids, each exactly once, using the marker <chart id="N" /> on its own line in the Key Findings section.
Allowed chart ids: ${idList}
Chart content:
${chartHints}

Do NOT invent chart ids outside the allowed set.`;
}

app.post("/api/generate", async (req, res) => {
  const { mlData, target_column, problem_type, customInstructions, allowedChartIds } = req.body || {};
  if (!mlData || !target_column || !problem_type) {
    return res.status(400).json({ error: "mlData, target_column, problem_type required" });
  }

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");

  try {
    // 4096 was too low for the full report (Exec Summary + 6 Key Findings paragraphs
    // + 3-subsection Dev Section with three pipe tables). When the limit is hit the
    // stream ends mid-sentence and the Developer Section gets silently dropped.
    const stream = await anthropic.messages.stream({
      model: CLAUDE_MODEL,
      max_tokens: 16000,
      messages: [
        {
          role: "user",
          content: buildNarrativePrompt({ mlData, target_column, problem_type, customInstructions, allowedChartIds }),
        },
      ],
    });

    let totalOut = 0;
    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
        res.write(event.delta.text);
        totalOut += event.delta.text.length;
      }
    }
    // Surface the stop reason so truncation bugs are obvious in the server logs.
    try {
      const finalMsg = await stream.finalMessage();
      console.log(
        `[generate] stop_reason=${finalMsg.stop_reason} out_chars=${totalOut} ` +
        `in_tokens=${finalMsg.usage?.input_tokens} out_tokens=${finalMsg.usage?.output_tokens}`,
      );
    } catch (_) {
      // non-fatal — already streamed to the client
    }
    res.end();
  } catch (e) {
    console.error("[generate] error:", e);
    if (!res.headersSent) res.status(500).json({ error: e.message });
    else res.end(`\n\n[stream error: ${e.message}]`);
  }
});

// ─── /api/export ───────────────────────────────────────────────────
function buildExportHtml({ markdown, mlData, chartImages, target_column, problem_type }) {
  // Replace every <chart id="N" /> marker with the matching image
  // (chartImages is in document order — same order the client captured).
  let imgIdx = 0;
  const replacedMd = (markdown || "").replace(/<chart\s+id=["']\d+["']\s*\/?>(\s*)/gi, () => {
    const src = chartImages?.[imgIdx++];
    return src
      ? `\n\n<figure class="chart-figure"><img src="${src}" alt="chart ${imgIdx}" /></figure>\n\n`
      : "";
  });

  // Minimal markdown-to-HTML — handles h1/h2/h3, paragraphs, lists, inline bold/em/code,
  // and GitHub-flavoured pipe tables. We deliberately avoid pulling in a markdown lib
  // server-side to keep deps light.
  const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function formatInline(s) {
    return s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  const splitRow = (line) => {
    // Strip leading/trailing pipes, split on interior pipes.
    const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").map((c) => c.trim());
  };

  const lines = replacedMd.split(/\r?\n/);
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimEnd();

    // ── Pipe table detection: current row + separator row (|---|---|…) ──
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const next = (lines[i + 1] || "").trimEnd();
      if (/^\s*\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(next)) {
        closeList();
        const header = splitRow(line);
        i += 2; // skip header + separator
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          rows.push(splitRow(lines[i]));
          i++;
        }
        i--; // outer loop will i++
        const isTotal = (r) => r.length && /^total$/i.test(r[0]);
        const bodyRows = rows.filter((r) => !isTotal(r));
        const totalRow = rows.find(isTotal);
        const thead = `<thead><tr>${header.map((h) => `<th>${formatInline(escape(h))}</th>`).join("")}</tr></thead>`;
        const tbody = `<tbody>${bodyRows
          .map((r) => `<tr>${r.map((c) => `<td>${formatInline(escape(c))}</td>`).join("")}</tr>`)
          .join("")}</tbody>`;
        const tfoot = totalRow
          ? `<tfoot><tr>${totalRow.map((c) => `<td>${formatInline(escape(c))}</td>`).join("")}</tr></tfoot>`
          : "";
        out.push(`<table>${thead}${tbody}${tfoot}</table>`);
        continue;
      }
    }

    if (line.startsWith("<figure")) { closeList(); out.push(line); continue; }
    if (/^### /.test(line)) { closeList(); out.push(`<h3>${escape(line.slice(4))}</h3>`); continue; }
    if (/^## /.test(line))  {
      closeList();
      const text = line.slice(3);
      // Developer Section MUST start on a new page.
      const isDev = /developer\s*section/i.test(text) || /^\s*3\./.test(text);
      out.push(`<h2${isDev ? ' class="page-break"' : ""}>${escape(text)}</h2>`);
      continue;
    }
    if (/^# /.test(line))   { closeList(); out.push(`<h1>${escape(line.slice(2))}</h1>`); continue; }
    if (/^[-*] /.test(line)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${formatInline(escape(line.slice(2)))}</li>`);
      continue;
    }
    closeList();
    if (line === "") out.push("");
    else out.push(`<p>${formatInline(escape(line))}</p>`);
  }
  closeList();
  const body = out.join("\n");

  // Document header (title + target subtitle) — matches the reference layout.
  const subtitleParts = [];
  if (target_column) subtitleParts.push(`Target: ${escape(String(target_column))}`);
  if (problem_type) subtitleParts[0] = `${subtitleParts[0] || "Target"} (${escape(String(problem_type))})`;
  const subtitle = subtitleParts.length
    ? `<p class="doc-subtitle">${subtitleParts.join("")}</p>`
    : "";

  return `<!doctype html>
<html><head><meta charset="utf-8"><title>Analysis Report</title>
<style>
  @page { size: A4; margin: 18mm 16mm; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #2A2A2A; font-size: 12px; line-height: 1.55; }
  h1 { font-size: 22px; border-bottom: 2px solid ${BRAND_ORANGE}; padding-bottom: 6px; margin: 0 0 14px; }
  h2 { font-size: 17px; border-bottom: 1px solid ${BRAND_ORANGE}; padding-bottom: 4px; margin: 22px 0 10px; }
  h2.page-break { page-break-before: always; break-before: page; margin-top: 0; }
  h3 { font-size: 14px; margin: 14px 0 6px; }
  p { margin: 0 0 10px; }
  ul { margin: 0 0 10px 20px; }
  strong { color: #1F1F1F; }
  /* Inline code references (column / identifier names from Claude) — plain
     orange text, no background box, no monospace, never breaks the line. */
  code { color: ${BRAND_ORANGE}; font-weight: 500; font-family: inherit; background: transparent; padding: 0; border-radius: 0; font-size: inherit; display: inline; }
  pre  { display: inline; margin: 0; padding: 0; background: transparent; font-family: inherit; }
  figure.chart-figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
  figure.chart-figure img { max-width: 100%; height: auto; border: 1px solid #E5E5E5; border-radius: 6px; }
  .doc-title { font-size: 24px; font-weight: 700; margin: 0 0 4px; color: #1F1F1F; }
  .doc-subtitle { font-size: 13px; color: #666; margin: 0 0 18px; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 11px; page-break-inside: avoid; }
  th { background: ${BRAND_ORANGE_LIGHT}; text-align: left; padding: 6px 8px; border-bottom: 1px solid #E5E5E5; font-weight: 600; }
  td { padding: 6px 8px; border-bottom: 1px solid #F0F0F0; vertical-align: top; }
  tfoot td { border-top: 2px solid ${BRAND_ORANGE}; font-weight: 600; background: ${BRAND_ORANGE_LIGHT}; }
</style></head>
<body>
<div class="doc-header">
  <div class="doc-title">Analysis Report</div>
  ${subtitle}
</div>
${body}
</body></html>`;
}

app.post("/api/export", async (req, res) => {
  const { markdown, mlData, chartImages, target_column, problem_type } = req.body || {};
  if (!markdown || !mlData) {
    return res.status(400).json({ error: "markdown and mlData required" });
  }

  let browser;
  try {
    const html = buildExportHtml({ markdown, mlData, chartImages, target_column, problem_type });
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });
    // Wait for any embedded chart figures to actually decode.
    await page.evaluate(async () => {
      const imgs = Array.from(document.querySelectorAll("figure.chart-figure img"));
      await Promise.all(imgs.map((img) => (img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; }))));
    });
    const pdf = await page.pdf({ format: "A4", printBackground: true });
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", `attachment; filename="analysis_report.pdf"`);
    res.end(pdf);
  } catch (e) {
    console.error("[export] error:", e);
    res.status(500).json({ error: e.message });
  } finally {
    if (browser) await browser.close();
  }
});

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`report-server listening on :${PORT}`);
});
