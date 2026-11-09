// Coordinator — refinement-loop agent. The hub: only it talks to the spokes,
// and only through tools whose inputs and outputs the code controls.

import type Anthropic from "@anthropic-ai/sdk";
import {
  MAX_NUDGES,
  MAX_STEPS,
  MAX_TOKENS,
  MAX_TOOL_ERRORS,
  MODEL,
} from "./config.ts";
import { checkOverlap, generatePartitions } from "./planner.ts";
import { createLoopState, type RunContext } from "./state.ts";
import { BAR, errMsg, indent } from "./text.ts";
import type { Tracer } from "./tracer.ts";
import type { RunResult } from "./types.ts";
import { documents } from "./untrusted.ts";
import { NUDGE_LOOP_OPEN, NUDGE_LOOP_STOPPED } from "../prompts/coordinator.ts";
import { tools } from "../tools/definitions.ts";
import { dispatchToolCalls } from "../tools/dispatch.ts";

export async function runCoordinator(
  tracer: Tracer,
  coordinatorPrompt: string,
  label: string,
  jobPosting: string,
  resume: string,
): Promise<RunResult> {
  console.log(`\n${BAR}\n${label}\n${BAR}`);
  console.log(`Run ${tracer.runId}  ·  trace: ${tracer.file}`);

  const ctx: RunContext = {
    tracer,
    jobPosting,
    resume,
    state: createLoopState(),
  };
  const { state } = ctx;
  let nudges = 0;
  let step = 0;

  const finish = (
    status: RunResult["status"],
    abortReason: string | null,
  ): RunResult => {
    const result: RunResult = {
      run_id: tracer.runId,
      trace_file: tracer.file,
      status,
      abort_reason: abortReason,
      stop_reason: state.stopReason,
      recommendation: state.recommendation,
      findings: state.findings,
      open_gaps:
        state.stopReason === "sufficient" ? [] : (state.lastEval?.gaps ?? []),
      interview_questions: state.lastEval?.interview_questions ?? [],
      delegated: state.delegated,
      steps: step,
      tool_errors: state.toolErrors,
    };
    if (abortReason) console.log(`\n[ABORTED] ${abortReason}`);
    tracer.event("run_end", {
      ...result,
      findings: state.findings.length,
      delegated: state.delegated.length,
      usage: Object.fromEntries(tracer.usage),
    });
    return result;
  };

  // Step 1 — generate non-overlapping partitions and show them to the human
  console.log("\nGenerating screening partitions...\n");
  let partitions;
  try {
    partitions = await generatePartitions(tracer, jobPosting, resume);
  } catch (err) {
    return finish("aborted", `partition planning failed: ${errMsg(err)}`);
  }
  state.partitionsByName = new Map(partitions.map((p) => [p.agent, p]));
  tracer.event("partitions", { partitions });
  console.log("Screening partitions:");
  console.log(JSON.stringify(partitions, null, 2));
  console.log();
  checkOverlap(tracer, partitions);

  console.log("Screening angles delegated:\n");

  // Step 2 — pass partitions to the coordinator so it routes correctly
  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content:
        `Please screen this application using the partitions below.\n\n` +
        `SCREENING PARTITIONS (non-overlapping scopes pre-planned for this candidate):\n` +
        `${JSON.stringify(partitions, null, 2)}\n\n` +
        documents(jobPosting, resume),
    },
  ];

  for (step = 1; step <= MAX_STEPS; step++) {
    let response: Anthropic.Message;
    try {
      response = await tracer.call(
        "coordinator",
        {
          model: MODEL,
          max_tokens: MAX_TOKENS.coordinator,
          system: coordinatorPrompt,
          tools,
          messages,
        },
        // Earlier messages are already in the trace as responses / tool_result events.
        {
          label: `step ${step}`,
          fromMessage: step === 1 ? 0 : messages.length,
        },
      );
    } catch (err) {
      return finish("aborted", `coordinator call failed: ${errMsg(err)}`);
    }

    for (const b of response.content) {
      if (b.type === "text" && b.text.trim()) {
        console.log(
          `  [coordinator] ${indent(b.text.trim(), "                ")}`,
        );
      }
    }

    if (response.stop_reason === "end_turn") {
      // A plain-text answer would bypass the code-enforced stop gate — send it back.
      if (++nudges > MAX_NUDGES) {
        return finish(
          "aborted",
          "coordinator kept ending its turn without submit_final",
        );
      }
      const nudge = state.stopReason ? NUDGE_LOOP_STOPPED : NUDGE_LOOP_OPEN;
      console.log(`  [NUDGE] ${nudge}`);
      tracer.event("coordinator_nudge", { step, message: nudge });
      messages.push(
        {
          role: "assistant",
          content: response.content.length ? response.content : "(no output)",
        },
        { role: "user", content: nudge },
      );
      continue;
    }

    if (response.stop_reason !== "tool_use") {
      // Without this guard, the same request would be re-sent until MAX_STEPS.
      return finish(
        "aborted",
        `unexpected coordinator stop_reason '${response.stop_reason}'`,
      );
    }

    const toolUses = response.content.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
    );
    const toolResults = await dispatchToolCalls(ctx, toolUses, step);
    messages.push(
      { role: "assistant", content: response.content },
      { role: "user", content: toolResults },
    );

    if (state.recommendation) return finish("submitted", null);
    if (state.toolErrors >= MAX_TOOL_ERRORS) {
      return finish(
        "aborted",
        `too many failed tool calls (${state.toolErrors})`,
      );
    }
  }

  step = MAX_STEPS;
  return finish("aborted", `reached step limit (${MAX_STEPS})`);
}
