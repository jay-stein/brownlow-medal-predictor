import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import TeamLogo from "./TeamLogo.jsx";
import { teamColor } from "./teams.js";

function RaceTooltip({ active, payload, label, players }) {
  if (!active || !payload?.length) return null;
  const rows = [...payload]
    .filter((entry) => entry.value !== undefined)
    .sort((a, b) => b.value - a.value);
  return (
    <div className="chart-tooltip">
      <div className="tooltip-title">{label}</div>
      {rows.map((entry) => {
        const player = players.find((candidate) => candidate.id === entry.dataKey);
        if (!player) return null;
        return (
          <div className="tooltip-row" key={entry.dataKey}>
            <span className="dot" style={{ background: teamColor(player.team) }} />
            {player.name}
            <b>{Number(entry.value).toFixed(1)}</b>
          </div>
        );
      })}
    </div>
  );
}

export default function TopWorms({ players, rounds, selectedId, onSelect, height = 320 }) {
  if (!players.length || !rounds?.length) return null;
  const data = rounds.map((round, index) => {
    const row = { round: round.label };
    players.forEach((player) => {
      row[player.id] = player.rounds.cumMean[index];
    });
    return row;
  });

  return (
    <div className="race-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="round" tick={{ fill: "#9fb0c3", fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fill: "#9fb0c3", fontSize: 11 }} />
          <Tooltip content={<RaceTooltip players={players} />} />
          {players.map((player) => (
            <Line
              key={player.id}
              type="monotone"
              dataKey={player.id}
              stroke={teamColor(player.team)}
              strokeWidth={player.id === selectedId ? 3 : 1.4}
              strokeOpacity={player.id === selectedId ? 1 : 0.5}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="race-legend">
        {players.map((player) => (
          <button
            type="button"
            key={player.id}
            className={player.id === selectedId ? "active" : ""}
            onClick={() => onSelect(player.id)}
          >
            <TeamLogo team={player.team} size={16} />
            {player.name}
            <small>{Number(player.expectedVotes ?? 0).toFixed(1)}</small>
          </button>
        ))}
      </div>
    </div>
  );
}
