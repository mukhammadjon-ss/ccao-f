import { BAR } from "../lib/text.ts";

const DIMENSIONS: [string, string[]][] = [
  [
    "technical / hard skill match",
    ["skill", "python", "fastapi", "keyword", "technical", "required"],
  ],
  [
    "experience depth / seniority",
    ["depth", "senior", "experience", "level", "scope"],
  ],
  [
    "red flags / disqualifiers",
    ["red flag", "gap", "hop", "disqualif", "concern"],
  ],
  [
    "soft skills / working style",
    ["soft", "communication", "collaborat", "working style", "team"],
  ],
  [
    "growth trajectory / potential",
    ["growth", "trajectory", "potential", "progression", "future"],
  ],
  [
    "nice-to-haves and trade-offs",
    ["nice", "trade", "bonus", "preferred", "optional"],
  ],
  [
    "compensation / motivation / timeline",
    ["compens", "salary", "motiv", "timeline", "expectation"],
  ],
  ["team / culture fit", ["culture", "team fit", "team dynamic", "environment"]],
  [
    "compensating strengths for gaps",
    ["compensat", "offset", "strength", "despite", "mitigat"],
  ],
];

/** Keyword heuristic over the delegated questions — a rough signal, not real coverage. */
export function coverageReport(dynamicQs: string[]): void {
  const combined = dynamicQs.join(" ").toLowerCase();
  const covered = (keywords: string[]) =>
    keywords.some((kw) => combined.includes(kw));

  console.log(
    `\n${BAR}\nCOVERAGE REPORT (keyword heuristic over questions)\n${BAR}`,
  );
  console.log(
    `\n  Dynamic coordinator:  ${String(dynamicQs.length).padStart(2)} screening angles\n`,
  );
  console.log(`  ${"Dimension".padEnd(45)} ${"Covered".padStart(8)}`);
  console.log(`  ${"-".repeat(45)} ${"-".repeat(8)}`);
  for (const [label, keywords] of DIMENSIONS) {
    console.log(
      `  ${label.padEnd(45)} ${(covered(keywords) ? "✓" : "✗").padStart(8)}`,
    );
  }
}
