import { useMemo, useRef, useState } from "react";
import { teamColor } from "./teams.js";

const PERIOD_FALLBACK = 1200;
const LATE_Q3_SECONDS = 1200;
const LAST10_SECONDS = 600;
const CLUTCH_MARGIN = 12;

const TYPE_LABEL = { G: "Goal", B: "Behind", R: "Rushed behind" };

function clock(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function isClutch(event) {
  return (
    event.type === "G" &&
    (event.p === 4 || (event.p === 3 && event.s >= LATE_Q3_SECONDS)) &&
    Math.abs(event.h - event.a) <= CLUTCH_MARGIN
  );
}

export default function MomentumWorm({ events, home, away }) {
  const [hovered, setHovered] = useState(null);
  const wrapper = useRef(null);

  const geometry = useMemo(() => {
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
    const eventPoints = [];
    for (const event of events) {
      const x = xAt(event.p, event.s);
      const previous = points[points.length - 1];
      points.push({ x, y: previous.y });
      const y = yAt(event.h - event.a);
      points.push({ x, y });
      eventPoints.push({ x, y });
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

    return {
      width,
      height,
      padX,
      padY,
      centre,
      maxima,
      lastPeriod,
      xAt,
      eventPoints,
      segments,
      homeColour,
      awayColour,
    };
  }, [events, home, away]);

  if (!geometry) return null;
  const { width, height, padX, padY, centre, maxima, xAt, eventPoints, segments } = geometry;

  const handlePointer = (clientX) => {
    const rect = wrapper.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const x = ((clientX - rect.left) / rect.width) * width;
    let nearest = 0;
    let best = Infinity;
    for (let index = 0; index < eventPoints.length; index += 1) {
      const distance = Math.abs(eventPoints[index].x - x);
      if (distance < best) {
        best = distance;
        nearest = index;
      }
    }
    setHovered((current) => (current === nearest ? current : nearest));
  };

  const hoveredEvent = hovered != null ? events[hovered] : null;
  const hoveredPoint = hovered != null ? eventPoints[hovered] : null;
  const tooltipLeft = hoveredPoint
    ? Math.min(86, Math.max(14, (hoveredPoint.x / width) * 100))
    : 50;

  let tooltip = null;
  if (hoveredEvent) {
    const margin = hoveredEvent.h - hoveredEvent.a;
    const leader = margin > 0 ? home : away;
    tooltip = (
      <div className="worm-tip" style={{ left: `${tooltipLeft}%` }}>
        <b>
          {TYPE_LABEL[hoveredEvent.type] ?? "Score"}
          {hoveredEvent.player ? ` · ${hoveredEvent.player}` : ""}
        </b>
        <span>
          Q{hoveredEvent.p} {clock(hoveredEvent.s)} · {home} {hoveredEvent.h}–{hoveredEvent.a}{" "}
          {away}
        </span>
        <span>
          {margin === 0 ? "level" : `${leader} by ${Math.abs(margin)}`}
          {isClutch(hoveredEvent) ? <em className="clutch"> · clutch goal</em> : ""}
        </span>
      </div>
    );
  }

  return (
    <div className="momentum-worm-wrap" ref={wrapper}>
      {tooltip}
      <svg
        className="momentum-worm"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Score margin through the match: ${home} versus ${away}. Hover for the scoreline at any moment.`}
        onPointerMove={(event) => handlePointer(event.clientX)}
        onPointerDown={(event) => handlePointer(event.clientX)}
        onPointerLeave={() => setHovered(null)}
        onPointerCancel={() => setHovered(null)}
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
        {[2, 3, 4].map((period) => (
          <line
            key={period}
            x1={xAt(period, 0)}
            x2={xAt(period, 0)}
            y1={padY - 3}
            y2={height - padY + 3}
            className="worm-quarter"
          />
        ))}
        {segments}
        {hoveredPoint ? (
          <line
            x1={hoveredPoint.x}
            x2={hoveredPoint.x}
            y1={padY - 3}
            y2={height - padY + 3}
            className="worm-cursor"
          />
        ) : null}
        {events.map((event, index) =>
          isClutch(event) ? (
            <circle
              key={`clutch-${index}`}
              cx={eventPoints[index].x}
              cy={eventPoints[index].y}
              r={3.2}
              className="worm-clutch"
            />
          ) : null
        )}
        {hoveredPoint ? (
          <circle
            cx={hoveredPoint.x}
            cy={hoveredPoint.y}
            r={4.6}
            className="worm-hover-dot"
          />
        ) : null}
      </svg>
    </div>
  );
}
