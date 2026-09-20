import useNarrow from "./useNarrow.js";
import { projectedPath, projectionBand } from "./countNight.js";
import { teamColor } from "./teams.js";

const PADDING = { left: 34, right: 16, top: 14, bottom: 24 };

export default function CountRace({ rows, roundIndex, roundList, calibration }) {
  const narrow = useNarrow();
  if (!rows.length) return null;

  const width = narrow ? 340 : 720;
  const height = narrow ? 210 : 250;
  const last = Math.max(roundList.length - 1, 1);

  const series = rows.map((row) => {
    const observed = row.entry.votes ?? 0;
    return {
      row,
      prior: row.player.rounds?.cumMean ?? [],
      path: projectedPath(row.player, roundIndex, observed),
      band: projectionBand(row.player, roundIndex, observed, calibration),
      color: teamColor(row.player.team),
    };
  });

  const values = [];
  for (const item of series) {
    item.prior.forEach((value) => value != null && values.push(value));
    item.band.upper.forEach((value) => value != null && values.push(value));
  }
  const yMax = Math.max(10, Math.ceil(Math.max(...values, 10) / 10) * 10);
  const x = (index) => PADDING.left + (index / last) * (width - PADDING.left - PADDING.right);
  const y = (value) => PADDING.top + (1 - value / yMax) * (height - PADDING.top - PADDING.bottom);

  const xTicks = [...new Set([0, 5, 10, 15, 20, last])].filter((index) => index <= last);
  const yTicks = [];
  for (let value = 0; value <= yMax; value += 10) yTicks.push(value);

  return (
    <div className="count-race">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Projected vote race for the selected players"
      >
        {yTicks.map((value) => (
          <g key={`y-${value}`}>
            <line
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={y(value)}
              y2={y(value)}
              className="race-grid"
            />
            <text x={PADDING.left - 6} y={y(value) + 3} className="race-label" textAnchor="end">
              {value}
            </text>
          </g>
        ))}
        {xTicks.map((index) => (
          <text
            key={`x-${index}`}
            x={x(index)}
            y={height - 7}
            className="race-label"
            textAnchor="middle"
          >
            {roundList[index]?.label ?? index}
          </text>
        ))}
        <line
          x1={x(roundIndex)}
          x2={x(roundIndex)}
          y1={PADDING.top}
          y2={height - PADDING.bottom}
          className="race-now"
        />
        {series.map((item, index) => (
          <polyline
            key={`prior-${index}`}
            points={item.prior.map((value, round) => `${x(round)},${y(value)}`).join(" ")}
            fill="none"
            stroke={item.color}
            strokeOpacity={0.28}
            strokeWidth={1.2}
          />
        ))}
        {series.map((item, index) => {
          const upper = item.band.upper
            .map((value, round) => (value == null ? null : [x(round), y(value)]))
            .filter(Boolean);
          const lower = item.band.lower
            .map((value, round) => (value == null ? null : [x(round), y(value)]))
            .filter(Boolean);
          if (!upper.length || !lower.length) return null;
          const polygon = upper.concat([...lower].reverse()).map((point) => point.join(",")).join(" ");
          return <polygon key={`band-${index}`} points={polygon} fill={item.color} opacity={0.1} />;
        })}
        {series.map((item, index) => {
          const points = item.path
            .map((value, round) => (value == null ? null : `${x(round)},${y(value)}`))
            .filter(Boolean)
            .join(" ");
          return (
            <polyline
              key={`path-${index}`}
              points={points}
              fill="none"
              stroke={item.color}
              strokeWidth={2.2}
              strokeDasharray="5 4"
            />
          );
        })}
        {series.map((item, index) => (
          <circle
            key={`dot-${index}`}
            cx={x(roundIndex)}
            cy={y(item.row.entry.votes ?? 0)}
            r={3.6}
            fill={item.color}
            stroke="rgba(0, 0, 0, 0.4)"
          />
        ))}
        {series.map((item, index) => {
          const final = item.path[Math.min(last, item.path.length - 1)];
          if (final == null) return null;
          const upper = item.band.upper[Math.min(last, item.band.upper.length - 1)] ?? final;
          const lower = item.band.lower[Math.min(last, item.band.lower.length - 1)] ?? final;
          return (
            <g key={`finish-${index}`}>
              <line x1={x(last)} x2={x(last)} y1={y(upper)} y2={y(lower)} stroke={item.color} strokeWidth={2} />
              <line x1={x(last) - 4} x2={x(last) + 4} y1={y(upper)} y2={y(upper)} stroke={item.color} strokeWidth={2} />
              <line x1={x(last) - 4} x2={x(last) + 4} y1={y(lower)} y2={y(lower)} stroke={item.color} strokeWidth={2} />
              <text x={x(last) - 7} y={y(final) - 6} className="race-value" textAnchor="end" fill={item.color}>
                {final.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="count-race-legend">
        {series.map((item, index) => (
          <span key={`legend-${index}`}>
            <i style={{ background: item.color }} />
            {item.row.player.name}
          </span>
        ))}
      </div>
    </div>
  );
}
