// Tool schema — screening spoke + refinement loop tools exposed to the coordinator.

import type Anthropic from "@anthropic-ai/sdk";

export const tools: Anthropic.Tool[] = [
  {
    name: "screening_agent",
    description:
      "Runs a specialist screening agent on one specific angle of candidate fit. " +
      "The system gives each call the job posting, the resume and the scope it must stay in. " +
      'In Phase 1, set "partition" to the partition agent name. ' +
      'In refinement, set "gap_index" to the index of the gap (from the latest ' +
      "evaluate_coverage result) this question fills. " +
      "A result starting with ERROR means the call failed and nothing was recorded.",
    input_schema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description: "The specific screening question to evaluate",
        },
        partition: {
          type: "string",
          description:
            "Phase 1 only: the partition agent name this call covers",
        },
        gap_index: {
          type: "integer",
          description:
            "Refinement only: index into the latest evaluate_coverage 'gaps' list",
        },
      },
      required: ["question"],
    },
  },
  {
    name: "evaluate_coverage",
    description:
      "Evaluate all screening findings collected so far (the system supplies them, " +
      "together with the job posting and resume). Call once after every partition has " +
      "been screened, and again after each refinement round. Returns a coverage score, " +
      "answerable gaps, interview-only questions, and loop_status (continue / stopped).",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "submit_final",
    description:
      "Submit the final HIRE / MAYBE / PASS recommendation. " +
      "Only accepted once evaluate_coverage has returned loop_status='stopped'.",
    input_schema: {
      type: "object",
      properties: {
        recommendation: {
          type: "string",
          description:
            "Final recommendation: HIRE, MAYBE, or PASS with full justification",
        },
      },
      required: ["recommendation"],
    },
  },
];

/** Execution order within one turn: screening → evaluate → submit. */
export const TOOL_ORDER: Record<string, number> = {
  screening_agent: 0,
  evaluate_coverage: 1,
  submit_final: 2,
};
