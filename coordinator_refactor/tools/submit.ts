// submit_final tool — only accepted once the code has stopped the loop.

import type { RunContext } from "../lib/state.ts";
import { ok, reject, type ToolResult } from "./results.ts";

export function handleSubmit(
  ctx: RunContext,
  id: string,
  input: Record<string, unknown>,
  turnHasEvaluate: boolean,
): ToolResult {
  const { state } = ctx;
  if (turnHasEvaluate) {
    return reject(
      id,
      "Wait for the evaluate_coverage result before calling submit_final.",
    );
  }
  if (state.evaluations === 0) return reject(id, "Run evaluate_coverage first.");
  if (!state.stopReason) {
    return reject(
      id,
      "The loop is still open: fill the latest gaps, then re-evaluate.",
    );
  }
  const text =
    typeof input.recommendation === "string"
      ? input.recommendation.trim()
      : "";
  if (!/\b(HIRE|MAYBE|PASS)\b/.test(text)) {
    return reject(
      id,
      "recommendation must state HIRE, MAYBE or PASS with justification.",
    );
  }
  state.recommendation = text;
  console.log(`\nFinal recommendation (${state.stopReason}):\n${text}`);
  return ok(id, "Recommendation submitted.");
}
