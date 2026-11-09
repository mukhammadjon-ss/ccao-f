import { stringList } from "./text.ts";
import type { CoverageEvaluation, Partition } from "./types.ts";

export function validatePartitions(raw: unknown): Partition[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error("expected a non-empty JSON array of partitions");
  }
  const names = new Set<string>();
  return raw.map((item, i) => {
    const p = (item ?? {}) as {
      agent?: unknown;
      scope?: Record<string, unknown>;
    };
    const agent = typeof p.agent === "string" ? p.agent.trim() : "";
    if (!agent) throw new Error(`partition ${i} has no "agent" name`);
    if (names.has(agent))
      throw new Error(`duplicate partition name '${agent}'`);
    names.add(agent);
    const topic = typeof p.scope?.topic === "string" ? p.scope.topic : "";
    const cover = stringList(p.scope?.cover);
    if (!topic || cover.length === 0) {
      throw new Error(
        `partition '${agent}' needs scope.topic and a non-empty scope.cover`,
      );
    }
    return {
      agent,
      scope: { topic, cover, exclude: stringList(p.scope?.exclude) },
    };
  });
}

export function validateEvaluation(raw: unknown): CoverageEvaluation {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("expected a JSON object");
  }
  const e = raw as Record<string, unknown>;
  const score = Number(e.coverage_score);
  if (!Number.isFinite(score)) throw new Error("coverage_score is missing");
  return {
    coverage_score: Math.max(0, Math.min(10, Math.round(score))),
    gaps: stringList(e.gaps),
    interview_questions: stringList(e.interview_questions),
    sufficient: e.sufficient === true,
    rationale: typeof e.rationale === "string" ? e.rationale : "",
  };
}
