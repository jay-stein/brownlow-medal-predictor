import { teamColor } from "./teams.js";

const PERIOD_FALLBACK = 1200;
const LATE_Q3_SECONDS = 1200;
const LAST10_SECONDS = 600;
const CLUTCH_MARGIN = 12;

function clock(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export default function MomentumWorm({ events, home, away }) {
  if (!events?.length) return null;

  const width = 320;
  const height = 60;
  const padX = 3;
  const padY = 7;
  const centre = height / 2;

  const maxima = { 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const event of events) {
    maxima[event.p] = Math.max(maxima[event.p] ?? 0, event.s);
  }
  const lastPeriod = Math.max(4, ...events.map((event) => event.p));
  const xAt = (period, seconds) => {
    const max = maxima[period] || PERIOD_FALLBACK;
    const fraction = period - 1 + Math.min(seconds / max, 1);
    return padX + (fraction / lastPeriod) * (width - 2 * padX);
  };
  const margins = events.map((event) => event.h - event.a);
  const maxAbs = Math.max(6, ...margins.map((margin) => Math.abs(margin)));
  const yAt = (margin) => centre - (margin / maxAbs) * (centre - padY);

  const points = [{ x: xAt(events[0].p, 0), y: centre }];
  for (const event of events) {
    const x = xAt(event.p, event.s);
    const previous = points[points.length - 1];
    points.push({ x, y: previous.y });
    points.push({ x, y: yAt(event.h - event.a) });
  }
  const tail = points[points.length - 1];
  points.push({ x: xAt(lastPeriod, maxima[lastPeriod] || PERIOD_FALLBACK), y: tail.y });

  const homeColour = teamColor(home);
  const awayColour = teamColor(away);
  const segments = [];
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    if (start.x === end.x && start.y === end.y) continue;
    segments.push(
      <line
        key={index}
        x1={start.x}
        y1={start.y}
        x2={end.x}
        y2={end.y}
        stroke={end.y <= centre ? homeColour : awayColour}
        strokeWidth={2}
        strokeLinecap="round"
      />
    );
  }

  const clutch = events.filter(
    (event) =>
      event.type === "G" &&
      (event.p === 4 || (event.p === 3 && event.s >= LATE_Q3_SECONDS)) &&
      Math.abs(event.h - event.a) <= CLUTCH_MARGIN
  );
  const quarterLines = [2, 3, 4].map((period) => (
    <line
      key={period}
      x1={xAt(period, 0)}
      x2={xAt(period, 0)}
      y1={padY - 3}
      y2={height - padY + 3}
      className="worm-quarter"
    />
  ));

  return (
    <svg
      className="momentum-worm"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Score margin through the match: ${home} versus ${away}`}
    >
      <rect
        x={xAt(4, 0)}
        y={padY - 3}
        width={Math.max(0, width - xAt(4, 0) - padX)}
        height={height - 2 * (padY - 3)}
        className="worm-q4"
      />
      {maxima[4] > LAST10_SECONDS ? (
        <rect
          x={xAt(4, LAST10_SECONDS)}
          y={padY - 3}
          width={Math.max(0, width - xAt(4, LAST10_SECONDS) - padX)}
          height={height - 2 * (padY - 3)}
          className="worm-last10"
        />
      ) : null}
      <line x1={padX} x2={width - padX} y1={centre} y2={centre} className="worm-zero" />
      {quarterLines}
      {segments}
      {clutch.map((event, index) => (
        <circle
          key={`clutch-${index}`}
          cx={xAt(event.p, event.s)}
          cy={yAt(event.h - event.a)}
          r={3.2}
          className="worm-clutch"
        >
          <title>
            {`${event.player ?? "Goal"} · Q${event.p} ${clock(event.s)} · margin ${
              event.h - event.a
            }`}
          </title>
        </circle>
      ))}
    </svg>
  );
}
