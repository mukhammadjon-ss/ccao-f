import { BAR } from "../lib/text.ts";
import type { Tracer } from "../lib/tracer.ts";
import type { RunResult, UsageTotals } from "../lib/types.ts";

/** Run outcome plus per-agent calls, errors, tokens and time. */
export function runSummary(result: RunResult, tracer: Tracer): void {
  console.log(`\n${BAR}\nRUN SUMMARY\n${BAR}`);
  console.log(`  Run:          ${result.run_id}`);
  console.log(
    `  Status:       ${result.status}${result.abort_reason ? ` (${result.abort_reason})` : ""}`,
  );
  console.log(`  Stop reason:  ${result.stop_reason ?? "-"}`);
  console.log(`  Steps:        ${result.steps}`);
  console.log(`  Findings:     ${result.findings.length}`);
  console.log(`  Tool errors:  ${result.tool_errors}`);
  if (result.open_gaps.length)
    console.log(`  Open gaps:    ${result.open_gaps.join(" | ")}`);

  console.log(
    `\n  ${"Agent".padEnd(18)} ${"calls".padStart(5)} ${"errors".padStart(6)} ` +
      `${"in tok".padStart(8)} ${"out tok".padStart(8)} ${"time".padStart(7)}`,
  );
  const total: UsageTotals = {
    calls: 0,
    errors: 0,
    input_tokens: 0,
    output_tokens: 0,
    ms: 0,
  };
  const row = (name: string, u: UsageTotals) =>
    console.log(
      `  ${name.padEnd(18)} ${String(u.calls).padStart(5)} ${String(u.errors).padStart(6)} ` +
        `${String(u.input_tokens).padStart(8)} ${String(u.output_tokens).padStart(8)} ` +
        `${(u.ms / 1000).toFixed(1).padStart(6)}s`,
    );
  for (const [role, u] of tracer.usage) {
    row(role, u);
    for (const k of Object.keys(total) as (keyof UsageTotals)[])
      total[k] += u[k];
  }
  row("TOTAL", total);
  console.log(`\n  Trace: ${result.trace_file}`);
}
