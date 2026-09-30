// dashboard.js
// Small shared helpers used across FairLoan pages.

function fairloanChartColors() {
  return {
    primary: "#2f5bea",
    accent: "#0fb99a",
    low: "#1a9c6b",
    moderate: "#b8860b",
    high: "#cf2b3f",
    gray: "#8b93ab",
  };
}

// Utility: format a 0-1 fraction as a percentage string.
function fmtPct(x, digits = 1) {
  return (x * 100).toFixed(digits) + "%";
}
