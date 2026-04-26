/**
 * CHART_REGISTRY — renderers keyed by chart id (1..6).
 *
 * Claude is the sole source of chart specs. The Python pipeline asks
 * the model to return an array of chart-config objects and stores the
 * array on `mlData.chartData`:
 *
 *   mlData.chartData = [
 *     {
 *       id: 1..6,
 *       title: "...",
 *       insight: "...",
 *       chartType: "BarChart" | "LineChart" | "AreaChart" | "PieChart"
 *                  | "ScatterChart" | "RadarChart" | "ComposedChart",
 *       xKey: "<field name on each row>",
 *       yKeys: ["<series field>", ...],
 *       data:  [{ <xKey>: ..., <yKey>: ..., ... }, ...],
 *       colors:["#RRGGBB", ...],        // one per yKey (optional)
 *     },
 *     ...
 *   ]
 *
 * <DynamicChart/> switches on chartType and renders the matching
 * Recharts component. Axis labels come from xKey / yKeys, and tooltips
 * always show the X value and every Y series — per the Python prompt
 * contract.
 */
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell, LabelList, LineChart, Line,
  AreaChart, Area, PieChart, Pie, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, ComposedChart,
} from "recharts";

const COLOR_PRIMARY = "#F97316";
const COLOR_DARK = "#EA580C";
const COLOR_LIGHT = "#FFF7ED";
const COLOR_RISK = "#D14343";
const COLOR_PROTECT = "#3F8F4D";
// Unified orange-family palette — used for EVERY chart regardless of what
// Claude returns in `config.colors`. Single-series charts get COLOR_PRIMARY;
// multi-series charts cycle through shades of orange so series stay
// distinguishable without breaking the single-colour brand look.
const ORANGE_PALETTE = [
  "#F97316", // primary
  "#EA580C", // darker
  "#FB923C", // lighter
  "#C2410C", // dark
  "#FDBA74", // pale
  "#9A3412", // deep
];

const tooltipStyle = {
  borderRadius: 8,
  border: `1px solid ${COLOR_PRIMARY}`,
  background: "#fff",
  fontSize: 12,
};
const axisTick = { fontSize: 12, fill: "#5C5C5C" };

// Axis-label helpers — consistent typography and placement across every chart.
// X labels sit below the tick area (use with sufficient bottom margin / XAxis height).
// Y labels are rotated -90° along the left of the plot area.
const xAxisLabel = (value) => ({
  value,
  position: "insideBottom",
  offset: -2,
  fontSize: 11,
  fill: "#3D3D3D",
});
const yAxisLabel = (value) => ({
  value,
  angle: -90,
  position: "insideLeft",
  offset: 10,
  fontSize: 11,
  fill: "#3D3D3D",
  style: { textAnchor: "middle" },
});

const truncate = (s, n = 18) => {
  const str = String(s ?? "");
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
};

// Convert raw field keys to readable axis titles.
//   "monthly_income"       → "Monthly Income"
//   "avgSessionDurationMs" → "Avg Session Duration Ms"
//   "OverTime"             → "Over Time"
const humanize = (s) => {
  if (s == null) return "";
  return String(s)
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
};

// Round to 2 decimals when value is a finite number — matches the prompt's
// "Decimal points should be standardized, rounded up to 2 decimal points" rule.
const fmtNum = (v) => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(2);
};

function ChartFrame({ title, subtitle, children, height = 280 }) {
  return (
    <div className="chart-block bg-white border border-gray-200 rounded-lg p-5 my-6">
      <h4 className="text-sm font-semibold text-text-primary mb-1">{title}</h4>
      {subtitle && <p className="text-xs text-text-secondary mb-3">{subtitle}</p>}
      <ResponsiveContainer width="100%" height={height}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}

// Per-yKey colour. We deliberately IGNORE `config.colors` from Claude —
// the user wants every chart to stay on the brand orange. Single series
// → primary orange; multi series → cycle through ORANGE_PALETTE.
// eslint-disable-next-line no-unused-vars
const seriesColor = (_colors, i) => ORANGE_PALETTE[i % ORANGE_PALETTE.length];

// Tooltip formatter that always exposes X + every Y value. Recharts
// already passes the x label via the `label` prop on <Tooltip>; we only
// need to format the series values here. Series name is humanised so
// tooltips read as "Monthly Income" rather than "monthly_income".
const tooltipFormatter = (value, name) => [fmtNum(value), humanize(name)];

// If Claude omitted yKeys, take every non-xKey numeric field on the first row.
function inferYKeys(data, xKey) {
  if (!Array.isArray(data) || data.length === 0) return [];
  const sample = data[0];
  return Object.keys(sample).filter(
    (k) => k !== xKey && typeof sample[k] !== "string" && sample[k] !== null,
  );
}

/**
 * DynamicChart — switches on `config.chartType` and renders the
 * matching Recharts component. Unknown/unsupported types fall back to
 * a BarChart so the slot still renders something useful.
 */
function DynamicChart({ config }) {
  if (!config || !Array.isArray(config.data) || config.data.length === 0) {
    return (
      <ChartFrame title={config?.title || "Chart"} subtitle={config?.insight || ""} height={220}>
        <div style={{ padding: 16, fontSize: 12, color: "#888" }}>
          No data provided for this chart.
        </div>
      </ChartFrame>
    );
  }

  const {
    title, insight, chartType, xKey, yKeys = [], data, colors,
  } = config;

  const ys = Array.isArray(yKeys) && yKeys.length > 0 ? yKeys : inferYKeys(data, xKey);
  const yLabel = ys.length === 1 ? humanize(ys[0]) : "Value";
  const xLabel = humanize(xKey);
  const kind = String(chartType || "").toLowerCase();
  const isSingle = ys.length === 1;

  // ── Pie ────────────────────────────────────────────────────────────
  if (kind === "piechart" || kind === "pie") {
    const yKey = ys[0];
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <PieChart margin={{ top: 10, right: 20, left: 20, bottom: 20 }}>
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v, n, p) => [fmtNum(v), p?.payload?.[xKey] ?? n]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Pie
            data={data}
            dataKey={yKey}
            nameKey={xKey}
            outerRadius={110}
            label={(d) => `${d[xKey]}: ${fmtNum(d[yKey])}`}
            labelLine={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={seriesColor(colors, i)} />
            ))}
          </Pie>
        </PieChart>
      </ChartFrame>
    );
  }

  // ── Radar ──────────────────────────────────────────────────────────
  if (kind === "radarchart" || kind === "radar") {
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <RadarChart data={data} margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
          <PolarGrid stroke="#E5E5E5" />
          <PolarAngleAxis dataKey={xKey} tick={axisTick} />
          <PolarRadiusAxis tick={axisTick} tickFormatter={fmtNum} />
          <Tooltip contentStyle={tooltipStyle} formatter={tooltipFormatter} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {ys.map((k, i) => (
            <Radar
              key={k}
              name={humanize(k)}
              dataKey={k}
              stroke={seriesColor(colors, i)}
              fill={seriesColor(colors, i)}
              fillOpacity={0.35}
            />
          ))}
        </RadarChart>
      </ChartFrame>
    );
  }

  // ── Scatter ────────────────────────────────────────────────────────
  if (kind === "scatterchart" || kind === "scatter") {
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <ScatterChart margin={{ top: 10, right: 20, left: 20, bottom: 35 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
          <XAxis
            type="number"
            dataKey={xKey}
            name={xLabel}
            tick={axisTick}
            tickFormatter={fmtNum}
            label={xAxisLabel(xLabel)}
          />
          <YAxis
            type="number"
            dataKey={ys[0]}
            name={humanize(ys[0])}
            tick={axisTick}
            tickFormatter={fmtNum}
            label={yAxisLabel(humanize(ys[0]))}
          />
          <ZAxis range={[50, 50]} />
          <Tooltip
            contentStyle={tooltipStyle}
            cursor={{ strokeDasharray: "3 3" }}
            formatter={(v, n) => [fmtNum(v), n]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {ys.map((k, i) => (
            <Scatter
              key={k}
              name={k}
              data={data.map((d) => ({ [xKey]: d[xKey], [k]: d[k] }))}
              fill={seriesColor(colors, i)}
              fillOpacity={0.75}
            />
          ))}
        </ScatterChart>
      </ChartFrame>
    );
  }

  // ── Area ───────────────────────────────────────────────────────────
  if (kind === "areachart" || kind === "area") {
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 35 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
          <XAxis
            dataKey={xKey}
            tick={axisTick}
            interval="preserveStartEnd"
            height={55}
            label={xAxisLabel(xLabel)}
          />
          <YAxis tick={axisTick} tickFormatter={fmtNum} label={yAxisLabel(yLabel)} />
          <Tooltip contentStyle={tooltipStyle} formatter={tooltipFormatter} />
          {!isSingle && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {ys.map((k, i) => (
            <Area
              key={k}
              type="monotone"
              dataKey={k}
              stroke={seriesColor(colors, i)}
              fill={seriesColor(colors, i)}
              fillOpacity={0.25}
              strokeWidth={2}
            />
          ))}
        </AreaChart>
      </ChartFrame>
    );
  }

  // ── Line ───────────────────────────────────────────────────────────
  if (kind === "linechart" || kind === "line") {
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 35 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
          <XAxis
            dataKey={xKey}
            tick={axisTick}
            interval="preserveStartEnd"
            height={55}
            label={xAxisLabel(xLabel)}
          />
          <YAxis tick={axisTick} tickFormatter={fmtNum} label={yAxisLabel(yLabel)} />
          <Tooltip contentStyle={tooltipStyle} formatter={tooltipFormatter} />
          {!isSingle && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {ys.map((k, i) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              stroke={seriesColor(colors, i)}
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ChartFrame>
    );
  }

  // ── Composed (mix of bars + lines when Claude asks for it) ─────────
  if (kind === "composedchart" || kind === "composed") {
    return (
      <ChartFrame title={title} subtitle={insight} height={320}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 35 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
          <XAxis dataKey={xKey} tick={axisTick} height={55} label={xAxisLabel(xLabel)} />
          <YAxis tick={axisTick} tickFormatter={fmtNum} label={yAxisLabel(yLabel)} />
          <Tooltip contentStyle={tooltipStyle} formatter={tooltipFormatter} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {ys.map((k, i) =>
            i % 2 === 0 ? (
              <Bar key={k} dataKey={k} fill={seriesColor(colors, i)} radius={[4, 4, 0, 0]} />
            ) : (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={seriesColor(colors, i)}
                strokeWidth={2}
                dot={{ r: 2 }}
              />
            ),
          )}
        </ComposedChart>
      </ChartFrame>
    );
  }

  // ── BarChart (default) ─────────────────────────────────────────────
  // Auto-rotate long x labels so category names don't clip.
  const needRotate = data.some((d) => String(d[xKey] ?? "").length > 8);
  return (
    <ChartFrame title={title} subtitle={insight} height={320}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: needRotate ? 40 : 30 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" vertical={false} />
        <XAxis
          dataKey={xKey}
          tick={axisTick}
          interval={0}
          angle={needRotate ? -15 : 0}
          textAnchor={needRotate ? "end" : "middle"}
          height={needRotate ? 75 : 40}
          label={xAxisLabel(xLabel)}
          tickFormatter={(v) => truncate(v, 18)}
        />
        <YAxis tick={axisTick} tickFormatter={fmtNum} label={yAxisLabel(yLabel)} />
        <Tooltip contentStyle={tooltipStyle} formatter={tooltipFormatter} />
        {!isSingle && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {ys.map((k, i) => (
          <Bar
            key={k}
            dataKey={k}
            fill={seriesColor(colors, i)}
            radius={[4, 4, 0, 0]}
          >
            {isSingle && (
              <LabelList
                dataKey={k}
                position="top"
                formatter={fmtNum}
                style={{ fontSize: 10, fill: "#3D3D3D" }}
              />
            )}
          </Bar>
        ))}
      </BarChart>
    </ChartFrame>
  );
}

// Kind strings we map to human-readable labels for the Visualizations picker.
const KIND_LABELS = {
  barchart: "Bar",
  linechart: "Line",
  areachart: "Area",
  piechart: "Pie",
  scatterchart: "Scatter",
  radarchart: "Radar",
  composedchart: "Composed",
};

function prettyKind(chartType) {
  return KIND_LABELS[String(chartType || "").toLowerCase()] || "Chart";
}

// ═════════════════════════════════════════════════════════════════════
//   CHART-SET SELECTOR
//   Claude returns an array of configs; each config's `id` maps to a
//   slot 1..6. Missing ids simply leave the slot empty.
// ═════════════════════════════════════════════════════════════════════

export function selectChartSet(mlData) {
  const chartArray = Array.isArray(mlData?.chartData) ? mlData.chartData : [];

  const byId = {};
  chartArray.forEach((cfg, i) => {
    if (!cfg || typeof cfg !== "object") return;
    const id = Number.isInteger(cfg.id) ? cfg.id : i + 1;
    byId[id] = cfg;
  });

  const registry = {};
  const meta = {};
  for (let id = 1; id <= 6; id++) {
    const cfg = byId[id];
    if (!cfg) continue;
    registry[id] = function ClaudeChart() {
      return <DynamicChart config={cfg} />;
    };
    meta[id] = { title: cfg.title || `Chart ${id}`, kind: prettyKind(cfg.chartType) };
  }
  return { registry, meta };
}

export function getChartRegistry(mlData) {
  return selectChartSet(mlData).registry;
}

export function chartMeta(id, mlData) {
  const { meta } = selectChartSet(mlData);
  return meta[id] ?? { title: `Chart ${id}`, kind: "" };
}

export const CHART_REGISTRY = new Proxy({}, {
  get() {
    throw new Error(
      "CHART_REGISTRY is now dynamic. Use getChartRegistry(mlData) or selectChartSet(mlData) instead."
    );
  },
});

export const REPORT_COLORS = { COLOR_PRIMARY, COLOR_DARK, COLOR_LIGHT, COLOR_RISK, COLOR_PROTECT };

// Exported so other pages/tests can render a single Claude-generated chart
// without going through the registry selector.
export { DynamicChart };
