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
  const [hoverFraction, setHoverFraction] = useState(null);
  const wrapper = useRef(null);

  const geometry = useMemo(() => {
    if (!events?.length) return null;
    const width = 320;
    const height = 60;
    const padX = 3;
    const padY = 7;
    const centre = height / 2;

    const maxima = {};
    for (const event of events) {
      maxima[event.p] = Math.max(maxima[event.p] ?? 0, event.s);
    }
    const lastPeriod = Math.max(4, ...events.map((event) => event.p));
    const toX = (fraction) => padX + (fraction / lastPeriod) * (width - 2 * padX);
    const toFraction = (period, seconds) => {
      const max = maxima[period] || PERIOD_FALLBACK;
      return period - 1 + Math.min(seconds / max, 1);
    };
    const eventFractions = events.map((event) => toFraction(event.p, event.s));

    const margins = events.map((event) => event.h - event.a);
    const maxAbs = Math.max(6, ...margins.map((margin) => Math.abs(margin)));
    const yAt = (margin) => centre - (margin / maxAbs) * (centre - padY);

    const points = [{ x: toX(0), y: centre }];
    for (const event of events) {
      const x = toX(toFraction(event.p, event.s));
      const previous = points[points.length - 1];
      points.push({ x, y: previous.y });
      points.push({ x, y: yAt(event.h - event.a) });
    }
    const tail = points[points.length - 1];
    points.push({ x: toX(lastPeriod), y: tail.y });

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
      toX,
      yAt,
      eventFractions,
      segments,
    };
  }, [events, home, away]);

  if (!geometry) return null;
  const { width, height, padX, padY, maxima, lastPeriod, toX, yAt, eventFractions, segments } =
    geometry;

  const handlePointer = (clientX) => {
    const rect = wrapper.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const x = ((clientX - rect.left) / rect.width) * width;
    const fraction = Math.min(
      lastPeriod,
      Math.max(0, ((x - padX) / (width - 2 * padX)) * lastPeriod)
    );
    setHoverFraction((current) => (current === fraction ? current : fraction));
  };

  let hover = null;
  if (hoverFraction != null) {
    const period = Math.min(lastPeriod, Math.floor(hoverFraction) + 1);
    const seconds = (hoverFraction - (period - 1)) * (maxima[period] || PERIOD_FALLBACK);
    let index = -1;
    for (let candidate = 0; candidate < eventFractions.length; candidate += 1) {
      if (eventFractions[candidate] <= hoverFraction) index = candidate;
    }
    const event = index >= 0 ? events[index] : null;
    const homeScore = event ? event.h : 0;
    const awayScore = event ? event.a : 0;
    const margin = homeScore - awayScore;
    hover = {
      period,
      seconds,
      x: toX(hoverFraction),
      y: yAt(margin),
      event,
      homeScore,
      awayScore,
      margin,
      index,
    };
  }

  const tooltipLeft = hover ? Math.min(86, Math.max(14, (hover.x / width) * 100)) : 50;
  let tooltip = null;
  if (hover) {
    tooltip = (
      <div className="worm-tip" style={{ left: `${tooltipLeft}%` }}>
        <b>
          {home} {hover.homeScore} vs {away} {hover.awayScore}
        </b>
        <span>
          Q{hover.period} {clock(hover.seconds)}
          {hover.event
            ? ` · ${TYPE_LABEL[hover.event.type] ?? "Score"}${
                hover.event.player ? ` · ${hover.event.player}` : ""
              }`
            : " · no score yet"}
          {hover.event && isClutch(hover.event) ? " · clutch" : ""}
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
        onPointerLeave={() => setHoverFraction(null)}
        onPointerCancel={() => setHoverFraction(null)}
      >
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          fill="transparent"
          pointerEvents="all"
        />
        <rect
          x={toX(3)}
          y={padY - 3}
          width={Math.max(0, width - toX(3) - padX)}
          height={height - 2 * (padY - 3)}
          className="worm-q4"
        />
        {maxima[4] > LAST10_SECONDS ? (
          <rect
            x={toX(3 + LAST10_SECONDS / maxima[4])}
            y={padY - 3}
            width={Math.max(0, width - toX(3 + LAST10_SECONDS / maxima[4]) - padX)}
            height={height - 2 * (padY - 3)}
            className="worm-last10"
          />
        ) : null}
        <line x1={padX} x2={width - padX} y1={geometry.centre} y2={geometry.centre} className="worm-zero" />
        {[2, 3, 4].map((period) => (
          <line
            key={period}
            x1={toX(period - 1)}
            x2={toX(period - 1)}
            y1={padY - 3}
            y2={height - padY + 3}
            className="worm-quarter"
          />
        ))}
        {segments}
        {hover ? (
          <line
            x1={hover.x}
            x2={hover.x}
            y1={padY - 3}
            y2={height - padY + 3}
            className="worm-cursor"
          />
        ) : null}
        {events.map((event, index) =>
          isClutch(event) ? (
            <circle
              key={`clutch-${index}`}
              cx={toX(eventFractions[index])}
              cy={yAt(event.h - event.a)}
              r={3.2}
              className="worm-clutch"
            />
          ) : null
        )}
        {hover ? (
          <circle cx={hover.x} cy={hover.y} r={4.4} className="worm-hover-dot" />
        ) : null}
      </svg>
    </div>
  );
}
