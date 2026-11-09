// Partition planner — generates non-overlapping screening scopes as JSON
// before any agent is invoked, so the coordinator can route without overlap.

import { MAX_TOKENS, MODEL, PLANNER_ATTEMPTS } from "./config.ts";
import { completeText, errMsg, normalize, parseJson } from "./text.ts";
import type { Tracer } from "./tracer.ts";
import type { Partition } from "./types.ts";
import { documents } from "./untrusted.ts";
import { validatePartitions } from "./validation.ts";
import { PARTITION_PLANNER_PROMPT } from "../prompts/planner.ts";

export async function generatePartitions(
  tracer: Tracer,
  jobPosting: string,
  resume: string,
): Promise<Partition[]> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= PLANNER_ATTEMPTS; attempt++) {
    try {
      const r = await tracer.call(
        "partition_planner",
        {
          model: MODEL,
          max_tokens: MAX_TOKENS.planner,
          system: PARTITION_PLANNER_PROMPT,
          messages: [
            {
              role: "user",
              content:
                `Generate screening partitions for this application.\n\n` +
                documents(jobPosting, resume),
            },
          ],
        },
        { label: `attempt ${attempt}` },
      );
      const text = completeText(tracer, r, "partition_planner");
      try {
        return validatePartitions(parseJson(text));
      } catch (err) {
        throw tracer.fail(
          "partition_planner",
          `invalid partitions: ${errMsg(err)}`,
        );
      }
    } catch (err) {
      lastError = err;
      if (attempt < PLANNER_ATTEMPTS) {
        console.log(`  [RETRY] partition planner (attempt ${attempt + 1})`);
      }
    }
  }
  throw lastError;
}

/** Warn (console + trace) when two partitions claim the same cover item. */
export function checkOverlap(tracer: Tracer, partitions: Partition[]): void {
  const seen = new Map<string, string>();
  for (const p of partitions) {
    for (const item of p.scope.cover) {
      const key = normalize(item);
      const owner = seen.get(key);
      if (owner && owner !== p.agent) {
        console.log(`  [OVERLAP WARNING] '${item}' in ${owner} and ${p.agent}`);
        tracer.event("partition_overlap", {
          item,
          partitions: [owner, p.agent],
        });
      }
      seen.set(key, p.agent);
    }
  }
}
