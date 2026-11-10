// ---------------------------------------------------------------------------
// Job Application Screener — hub-and-spoke multi-agent pipeline with a
// code-enforced refinement loop and a full observability layer.
//
//   data/     sample job posting + resume
//   lib/      config, types, tracer, coordinator loop, planner, helpers
//   prompts/  system prompts for every agent
//   tools/    coordinator tool schemas and their handlers (spokes)
//   reports/  console reports printed after the run
// ---------------------------------------------------------------------------

import Anthropic from "@anthropic-ai/sdk";
import { candidateLabel, jobPosting, resume } from "./data/application.ts";
import {
  MAX_REFINEMENTS,
  MAX_STEPS,
  MAX_TOOL_ERRORS,
  MODEL,
} from "./lib/config.ts";
import { runCoordinator } from "./lib/coordinator.ts";
import { Tracer } from "./lib/tracer.ts";
import { DYNAMIC_COORDINATOR } from "./prompts/coordinator.ts";
import { coverageReport } from "./reports/coverage.ts";
import { runSummary } from "./reports/summary.ts";

process.loadEnvFile();

async function main(): Promise<void> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const tracer = new Tracer(client);
  tracer.event("run_start", {
    model: MODEL,
    max_steps: MAX_STEPS,
    max_refinements: MAX_REFINEMENTS,
    max_tool_errors: MAX_TOOL_ERRORS,
  });

  console.log(`Candidate: ${candidateLabel}\n`);
  console.log(
    "The coordinator reads the candidate first, then routes to relevant checks only.",
  );

  const result = await runCoordinator(
    tracer,
    DYNAMIC_COORDINATOR,
    "DYNAMIC COORDINATOR  (reads candidate first, routes to relevant checks only)",
    jobPosting,
    resume,
  );

  coverageReport(result.delegated);
  runSummary(result, tracer);
  if (result.status !== "submitted") process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
