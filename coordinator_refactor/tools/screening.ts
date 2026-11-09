// screening_agent tool — validates the coordinator's call, then runs the spoke.

import { MAX_TOKENS, MODEL } from "../lib/config.ts";
import type { RunContext } from "../lib/state.ts";
import { completeText, errMsg, indent } from "../lib/text.ts";
import type { Partition } from "../lib/types.ts";
import { documents, tag } from "../lib/untrusted.ts";
import { SCREENING_AGENT_PROMPT } from "../prompts/screening.ts";
import { ok, reject, toolError, type ToolResult } from "./results.ts";

export interface ScreeningJob {
  kind: "job";
  id: string;
  source: string;
  question: string;
  scope: string;
  /** Undo the reservation if the spoke call fails, so it can be retried. */
  release: () => void;
}

const partitionScope = (p: Partition) =>
  `Partition: ${p.agent}\nTopic: ${p.scope.topic}\n` +
  `In scope: ${p.scope.cover.join("; ")}\n` +
  `Out of scope: ${p.scope.exclude.join("; ") || "(none)"}`;

const gapScope = (gap: string) =>
  `Fill this specific gap left by earlier screening, and nothing else: ${gap}`;

/** Validate and reserve a screening call. Synchronous, so parallel calls can't race. */
export function prepareScreening(
  ctx: RunContext,
  id: string,
  input: Record<string, unknown>,
): ScreeningJob | ToolResult {
  const { state } = ctx;
  if (state.stopReason) {
    return reject(
      id,
      `The loop has stopped (${state.stopReason}). Call submit_final.`,
    );
  }
  const question =
    typeof input.question === "string" ? input.question.trim() : "";
  if (!question) return reject(id, "question is required.");

  if (state.evaluations === 0) {
    // Phase 1 — exactly one call per known partition
    const name = input.partition;
    const partition =
      typeof name === "string" ? state.partitionsByName.get(name) : undefined;
    if (!partition) {
      return reject(
        id,
        `Phase 1 calls must set "partition" to one of: ${[...state.partitionsByName.keys()].join(", ")}.`,
      );
    }
    if (state.screenedPartitions.has(partition.agent)) {
      return reject(
        id,
        `Partition '${partition.agent}' has already been screened.`,
      );
    }
    state.screenedPartitions.add(partition.agent);
    return {
      kind: "job",
      id,
      source: `partition:${partition.agent}`,
      question,
      scope: partitionScope(partition),
      release: () => state.screenedPartitions.delete(partition.agent),
    };
  }

  // Refinement — every call must target an open gap from the latest evaluation
  const gaps = state.lastEval!.gaps;
  const idx = input.gap_index;
  if (
    typeof idx !== "number" ||
    !Number.isInteger(idx) ||
    idx < 0 ||
    idx >= gaps.length
  ) {
    return reject(
      id,
      `Refinement calls must set "gap_index" to 0-${gaps.length - 1} from the latest gaps list.`,
    );
  }
  if (state.filledGaps.has(idx)) {
    return reject(id, `Gap ${idx} has already been addressed this round.`);
  }
  state.filledGaps.add(idx);
  return {
    kind: "job",
    id,
    source: `gap:${gaps[idx]}`,
    question,
    scope: gapScope(gaps[idx]),
    release: () => state.filledGaps.delete(idx),
  };
}

/** The spoke itself: stateless, no tools, input built entirely by the code. */
async function callScreeningAgent(
  ctx: RunContext,
  job: ScreeningJob,
): Promise<string> {
  const r = await ctx.tracer.call(
    "screening_agent",
    {
      model: MODEL,
      max_tokens: MAX_TOKENS.screening,
      system: SCREENING_AGENT_PROMPT,
      messages: [
        {
          role: "user",
          content:
            `QUESTION: ${job.question}\n\n` +
            `${tag("scope", job.scope)}\n\n` +
            documents(ctx.jobPosting, ctx.resume),
        },
      ],
    },
    { label: job.source },
  );
  return completeText(ctx.tracer, r, "screening_agent");
}

/** Run reserved screening calls in parallel; record findings in call order. */
export async function runScreening(
  ctx: RunContext,
  jobs: ScreeningJob[],
  results: Map<string, ToolResult>,
): Promise<void> {
  const { state } = ctx;
  for (const j of jobs) console.log(`  [→] (${j.source}) ${j.question}`);
  const settled = await Promise.allSettled(
    jobs.map((j) => callScreeningAgent(ctx, j)),
  );
  jobs.forEach((j, i) => {
    const s = settled[i];
    if (s.status === "fulfilled") {
      state.findings.push({
        source: j.source,
        question: j.question,
        answer: s.value,
      });
      state.delegated.push(j.question);
      console.log(
        `  [${String(state.findings.length).padStart(2)}] ← (${j.source})\n        ${indent(s.value)}`,
      );
      results.set(j.id, ok(j.id, s.value));
    } else {
      j.release();
      results.set(
        j.id,
        toolError(
          ctx,
          j.id,
          `screening_agent failed (${j.source}): ${errMsg(s.reason)}`,
        ),
      );
    }
  });
}
