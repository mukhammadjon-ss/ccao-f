import type { Tracer } from "./tracer.ts";
import type {
  CoverageEvaluation,
  Finding,
  Partition,
  StopReason,
} from "./types.ts";

/** Loop state, owned by the code — the coordinator only sees tool results. */
export interface LoopState {
  delegated: string[];
  findings: Finding[];
  partitionsByName: Map<string, Partition>;
  screenedPartitions: Set<string>;
  /** Successful evaluations: 1 = initial, 2+ = after refinement. */
  evaluations: number;
  lastEval: CoverageEvaluation | null;
  findingsAtLastEval: number;
  /** Gap indices filled since the last evaluation. */
  filledGaps: Set<number>;
  stopReason: StopReason | null;
  recommendation: string | null;
  toolErrors: number;
}

/** Everything a tool handler needs: the tracer, the documents and the loop state. */
export interface RunContext {
  tracer: Tracer;
  jobPosting: string;
  resume: string;
  state: LoopState;
}

export function createLoopState(partitions: Partition[] = []): LoopState {
  return {
    delegated: [],
    findings: [],
    partitionsByName: new Map(partitions.map((p) => [p.agent, p])),
    screenedPartitions: new Set(),
    evaluations: 0,
    lastEval: null,
    findingsAtLastEval: 0,
    filledGaps: new Set(),
    stopReason: null,
    recommendation: null,
    toolErrors: 0,
  };
}
