import path from "node:path";
import { fileURLToPath } from "node:url";

export const MODEL = "claude-haiku-5-5";
// Haiku 5.5 thinks by default and thinking counts against max_tokens,
// so every call leaves generous headroom for the visible answer.

/** Traces are written to coordinator_refactor/logs/<run_id>.jsonl. */
export const LOG_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "logs",
);

export const MAX_STEPS = 30;
/** Refinement rounds AFTER the initial evaluation. */
export const MAX_REFINEMENTS = 3;
/** Failed spoke/evaluator calls tolerated before the run is aborted. */
export const MAX_TOOL_ERRORS = 5;
/** Times the coordinator may end its turn without submit_final before we abort. */
export const MAX_NUDGES = 2;
export const PLANNER_ATTEMPTS = 2;

export const MAX_TOKENS = {
  planner: 4096,
  screening: 2048,
  evaluation: 4096,
  coordinator: 8192,
} as const;
