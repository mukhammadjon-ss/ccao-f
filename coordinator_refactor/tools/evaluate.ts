// evaluate_coverage tool — runs the evaluator spoke and makes the stop decision in code.

import { MAX_REFINEMENTS, MAX_TOKENS, MODEL } from "../lib/config.ts";
import type { RunContext } from "../lib/state.ts";
import { completeText, errMsg, normalize, parseJson } from "../lib/text.ts";
import type { CoverageEvaluation, Finding, StopReason } from "../lib/types.ts";
import { documents, tag } from "../lib/untrusted.ts";
import { validateEvaluation } from "../lib/validation.ts";
import { EVALUATION_AGENT_PROMPT } from "../prompts/evaluation.ts";
import { ok, reject, toolError, type ToolResult } from "./results.ts";

function formatFindings(findings: Finding[]): string {
  return findings
    .map(
      (f, i) =>
        `${i + 1}. [${f.source}]\n   Q: ${f.question}\n   A: ${f.answer}`,
    )
    .join("\n\n");
}

/** The evaluator sees the source documents plus the code-held findings — not a paraphrase. */
async function callEvaluationAgent(
  ctx: RunContext,
  round: number,
): Promise<CoverageEvaluation> {
  const r = await ctx.tracer.call(
    "evaluation_agent",
    {
      model: MODEL,
      max_tokens: MAX_TOKENS.evaluation,
      system: EVALUATION_AGENT_PROMPT,
      messages: [
        {
          role: "user",
          content:
            `${documents(ctx.jobPosting, ctx.resume)}\n\n` +
            tag("findings", formatFindings(ctx.state.findings)),
        },
      ],
    },
    { label: `round ${round}` },
  );
  const text = completeText(ctx.tracer, r, "evaluation_agent");
  try {
    return validateEvaluation(parseJson(text));
  } catch (err) {
    throw ctx.tracer.fail(
      "evaluation_agent",
      `invalid evaluation: ${errMsg(err)}`,
    );
  }
}

/** Stop conditions — decided by code, not by the model. */
function decideStop(
  evaluation: CoverageEvaluation,
  previous: CoverageEvaluation | null,
  round: number,
): StopReason | null {
  if (evaluation.sufficient) return "sufficient";
  if (evaluation.gaps.length === 0) return "no_answerable_gaps";
  if (round >= MAX_REFINEMENTS) return "limit_reached";
  if (previous) {
    const scoreImproved = evaluation.coverage_score > previous.coverage_score;
    const prevGaps = new Set(previous.gaps.map(normalize));
    const allGapsRepeated = evaluation.gaps.every((g) =>
      prevGaps.has(normalize(g)),
    );
    if (!scoreImproved || allGapsRepeated) return "no_progress";
  }
  return null;
}

export async function handleEvaluate(
  ctx: RunContext,
  id: string,
): Promise<ToolResult> {
  const { state, tracer } = ctx;
  if (state.stopReason) {
    return reject(
      id,
      `The loop has stopped (${state.stopReason}). Call submit_final.`,
    );
  }
  if (
    state.evaluations === 0 &&
    state.screenedPartitions.size < state.partitionsByName.size
  ) {
    const missing = [...state.partitionsByName.keys()].filter(
      (n) => !state.screenedPartitions.has(n),
    );
    return reject(
      id,
      `Screen every partition first. Missing: ${missing.join(", ")}.`,
    );
  }
  if (
    state.evaluations > 0 &&
    state.findings.length === state.findingsAtLastEval
  ) {
    return reject(
      id,
      "No new findings since the last evaluation — fill the gaps first.",
    );
  }

  const round = state.evaluations; // 0 = initial evaluation
  console.log(
    round === 0
      ? "\n[Evaluating coverage — initial]"
      : `\n[Evaluating coverage — refinement round ${round}/${MAX_REFINEMENTS}]`,
  );

  let evaluation: CoverageEvaluation;
  try {
    evaluation = await callEvaluationAgent(ctx, round);
  } catch (err) {
    // State untouched, so the coordinator can simply call evaluate_coverage again.
    return toolError(ctx, id, `evaluate_coverage failed: ${errMsg(err)}`);
  }
  state.evaluations++;

  const { gaps } = evaluation;
  console.log(
    `  Coverage score: ${evaluation.coverage_score}/10 — ${evaluation.rationale}`,
  );
  console.log(
    gaps.length ? `  Gaps: ${gaps.join(" | ")}` : "  No answerable gaps.",
  );
  if (evaluation.interview_questions.length) {
    console.log(
      `  Interview-only: ${evaluation.interview_questions.join(" | ")}`,
    );
  }
  console.log(`  Sufficient: ${evaluation.sufficient}`);

  state.stopReason = decideStop(evaluation, state.lastEval, round);
  if (state.stopReason) console.log(`  [Loop stopped: ${state.stopReason}]`);

  tracer.event("loop_decision", {
    round,
    coverage_score: evaluation.coverage_score,
    previous_score: state.lastEval?.coverage_score ?? null,
    sufficient: evaluation.sufficient,
    gaps,
    interview_questions: evaluation.interview_questions,
    findings: state.findings.length,
    loop_status: state.stopReason ? "stopped" : "continue",
    stop_reason: state.stopReason,
  });

  state.lastEval = evaluation;
  state.findingsAtLastEval = state.findings.length;
  state.filledGaps.clear();

  // Report the real evaluation; never fake sufficient=true
  return ok(
    id,
    JSON.stringify({
      ...evaluation,
      refinement_round: round,
      max_refinements: MAX_REFINEMENTS,
      loop_status: state.stopReason ? "stopped" : "continue",
      stop_reason: state.stopReason,
      remaining_gaps:
        state.stopReason && state.stopReason !== "sufficient" ? gaps : [],
    }),
  );
}
