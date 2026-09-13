import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function WormTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-title">{label}</div>
      <div className="tooltip-row">
        <span>Median</span>
        <b>{row.cumMedian}</b>
      </div>
      <div className="tooltip-row">
        <span>Mean</span>
        <b>{row.cumMean}</b>
      </div>
      <div className="tooltip-row">
        <span>50% band</span>
        <b>
          {row.cumQ25} – {row.cumQ75}
        </b>
      </div>
      <div className="tooltip-row">
        <span>90% band</span>
        <b>
          {row.cumQ05} – {row.cumQ95}
        </b>
      </div>
    </div>
  );
}

export default function WormChart({ data, player, showPaths, height = 520 }) {
  const pathKeys = player?.rounds?.paths?.map((_, index) => `path_${index}`) ?? [];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
        <defs>
          <linearGradient id="outerBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f6e27a" stopOpacity={0.22} />
            <stop offset="100%" stopColor="#d4af37" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="innerBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f6e27a" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#c39a1f" stopOpacity={0.16} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(212,175,55,0.12)" vertical={false} />
        <XAxis
          dataKey="round"
          interval={2}
          tick={{ fill: "#9aa4bf", fontSize: 12 }}
          tickLine={false}
          axisLine={{ stroke: "rgba(212,175,55,0.25)" }}
        />
        <YAxis
          tick={{ fill: "#9aa4bf", fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          width={44}
        />
        <Tooltip
          content={<WormTooltip />}
          cursor={{ stroke: "rgba(246,226,122,0.35)", strokeDasharray: "4 4" }}
        />
        {showPaths
          ? pathKeys.map((key) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                dot={false}
                stroke="#f6e27a"
                strokeOpacity={0.12}
                strokeWidth={1}
                isAnimationActive={false}
              />
            ))
          : null}
        <Area
          type="monotone"
          dataKey="cumQ05"
          stackId="outer"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="cumOuter"
          stackId="outer"
          stroke="none"
          fill="url(#outerBand)"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="cumQ25"
          stackId="inner"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="cumInner"
          stackId="inner"
          stroke="none"
          fill="url(#innerBand)"
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="cumMean"
          stroke="#8ad1ff"
          strokeDasharray="5 5"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="cumMedian"
          stroke="#ffe9a3"
          strokeWidth={3}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
