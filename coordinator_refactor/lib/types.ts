export interface PartitionScope {
  topic: string;
  cover: string[];
  exclude: string[];
}

export interface Partition {
  agent: string;
  scope: PartitionScope;
}

export interface CoverageEvaluation {
  coverage_score: number;
  /** Gaps answerable from the job posting + resume — the only ones refinement can fill. */
  gaps: string[];
  /** Things that matter but can't be answered from the documents. */
  interview_questions: string[];
  sufficient: boolean;
  rationale: string;
}

/** One screening result, stored by the code (not just in the coordinator's context). */
export interface Finding {
  source: string; // "partition:<name>" or "gap:<text>"
  question: string;
  answer: string;
}

export type StopReason =
  | "sufficient"
  | "limit_reached"
  | "no_progress"
  | "no_answerable_gaps";

export type AgentRole =
  | "partition_planner"
  | "screening_agent"
  | "evaluation_agent"
  | "coordinator";

export interface UsageTotals {
  calls: number;
  errors: number;
  input_tokens: number;
  output_tokens: number;
  ms: number;
}

export interface RunResult {
  run_id: string;
  trace_file: string;
  status: "submitted" | "aborted";
  abort_reason: string | null;
  stop_reason: StopReason | null;
  recommendation: string | null;
  findings: Finding[];
  open_gaps: string[];
  interview_questions: string[];
  delegated: string[];
  steps: number;
  tool_errors: number;
}
