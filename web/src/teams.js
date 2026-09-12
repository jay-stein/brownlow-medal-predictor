export const TEAM_COLORS = {
  "Adelaide Crows": "#e2b13c",
  "Brisbane Lions": "#a30046",
  Carlton: "#3f5d8f",
  Collingwood: "#e8e8e8",
  Essendon: "#cc0000",
  Fremantle: "#7a4fb4",
  "Geelong Cats": "#5f8fd0",
  "Gold Coast SUNS": "#d93a00",
  "GWS GIANTS": "#f58022",
  Hawthorn: "#b8873f",
  Melbourne: "#3d6fd1",
  "North Melbourne": "#4f9fe0",
  "Port Adelaide": "#00a7b5",
  Richmond: "#ffd200",
  "St Kilda": "#ed1b2e",
  "Sydney Swans": "#e4002b",
  "West Coast Eagles": "#f2c14e",
  "Western Bulldogs": "#2f7fd1",
};

export function teamColor(team) {
  return TEAM_COLORS[team] ?? "#d4af37";
}
