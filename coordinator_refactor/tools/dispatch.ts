// Runs one coordinator turn's tool calls in a safe order and traces each one.

import type Anthropic from "@anthropic-ai/sdk";
import type { RunContext } from "../lib/state.ts";
import { TOOL_ORDER } from "./definitions.ts";
import { handleEvaluate } from "./evaluate.ts";
import { reject, type ToolResult } from "./results.ts";
import { prepareScreening, runScreening, type ScreeningJob } from "./screening.ts";
import { handleSubmit } from "./submit.ts";

/**
 * Screening first (in parallel), then evaluation, then submit — so an evaluation
 * in the same turn sees that turn's findings, and submit never runs ahead of it.
 * Results are returned in the coordinator's original call order.
 */
export async function dispatchToolCalls(
  ctx: RunContext,
  toolUses: Anthropic.ToolUseBlock[],
  step: number,
): Promise<ToolResult[]> {
  const { tracer } = ctx;
  for (const b of toolUses) {
    tracer.event("tool_call", {
      step,
      tool_use_id: b.id,
      name: b.name,
      input: b.input,
    });
  }

  const turnHasEvaluate = toolUses.some((b) => b.name === "evaluate_coverage");
  const sorted = [...toolUses].sort(
    (a, b) => (TOOL_ORDER[a.name] ?? 3) - (TOOL_ORDER[b.name] ?? 3),
  );
  const results = new Map<string, ToolResult>();
  const inputOf = (b: Anthropic.ToolUseBlock) =>
    (b.input ?? {}) as Record<string, unknown>;

  const jobs: ScreeningJob[] = [];
  for (const block of sorted.filter((b) => b.name === "screening_agent")) {
    const prepared = prepareScreening(ctx, block.id, inputOf(block));
    if ("kind" in prepared) jobs.push(prepared);
    else results.set(block.id, prepared);
  }
  if (jobs.length) await runScreening(ctx, jobs, results);

  for (const block of sorted.filter((b) => b.name !== "screening_agent")) {
    if (block.name === "evaluate_coverage") {
      results.set(block.id, await handleEvaluate(ctx, block.id));
    } else if (block.name === "submit_final") {
      results.set(
        block.id,
        handleSubmit(ctx, block.id, inputOf(block), turnHasEvaluate),
      );
    } else {
      results.set(block.id, reject(block.id, `Unknown tool '${block.name}'.`));
    }
  }

  const toolResults = toolUses.map((b) => results.get(b.id)!);
  for (const r of toolResults) {
    tracer.event("tool_result", {
      step,
      tool_use_id: r.tool_use_id,
      is_error: r.is_error ?? false,
      content: r.content,
    });
  }
  return toolResults;
}
