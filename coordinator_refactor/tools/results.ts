import type Anthropic from "@anthropic-ai/sdk";
import type { RunContext } from "../lib/state.ts";

export type ToolResult = Anthropic.ToolResultBlockParam;

export const ok = (id: string, content: string): ToolResult => ({
  type: "tool_result",
  tool_use_id: id,
  content,
});

/** The coordinator broke a rule — tell it what to correct. */
export const reject = (id: string, msg: string): ToolResult => {
  console.log(`  [REJECTED] ${msg}`);
  return {
    type: "tool_result",
    tool_use_id: id,
    content: `REJECTED: ${msg}`,
    is_error: true,
  };
};

/** A spoke/evaluator call failed — counts against the run's error budget. */
export const toolError = (
  ctx: RunContext,
  id: string,
  msg: string,
): ToolResult => {
  ctx.state.toolErrors++;
  return {
    type: "tool_result",
    tool_use_id: id,
    content: `ERROR: ${msg}. Nothing was recorded; you may retry this call.`,
    is_error: true,
  };
};
