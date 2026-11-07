import Anthropic from "@anthropic-ai/sdk";

process.loadEnvFile();
// ---------------------------------------------------------------------------
// Use case: Job Application Screener — hub-and-spoke multi-agent pipeline
// with a code-enforced refinement loop.
// ---------------------------------------------------------------------------

const MODEL = "claude-haiku-5-5";
// Haiku 5.5 thinks by default and thinking counts against max_tokens,
// so every call below leaves generous headroom for the visible answer.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PartitionScope {
  topic: string;
  cover: string[];
  exclude: string[];
}

interface Partition {
  agent: string;
  scope: PartitionScope;
}

interface CoverageEvaluation {
  coverage_score: number;
  /** Gaps answerable from the job posting + resume — the only ones refinement can fill. */
  gaps: string[];
  /** Things that matter but can't be answered from the documents. */
  interview_questions: string[];
  sufficient: boolean;
  rationale: string;
}

/** One screening result, stored by the code (not just in the coordinator's context). */
interface Finding {
  source: string; // "partition:<name>" or "gap:<text>"
  question: string;
  answer: string;
}

type StopReason =
  | "sufficient"
  | "limit_reached"
  | "no_progress"
  | "no_answerable_gaps";

// ---------------------------------------------------------------------------
// Spoke system prompts
// ---------------------------------------------------------------------------

const SCREENING_AGENT_PROMPT = `You are a specialist hiring analyst. You will be given a specific
screening question about a candidate along with the job posting and resume.

Answer the question in 2-3 focused sentences. Be concrete and specific to this candidate.`;

const EVALUATION_AGENT_PROMPT = `You are a screening coverage evaluator. You receive the job posting,
the resume, and the screening findings collected so far.

Decide what is still missing for a confident HIRE / MAYBE / PASS decision on THIS candidate
for THIS role. Split missing items into two lists:
  - "gaps": questions that CAN be answered from the job posting and resume but are not yet
    answered by the findings. Only these can be filled by further screening.
  - "interview_questions": things that matter but CANNOT be answered from the documents
    (e.g. salary expectations, culture fit, references, availability). Never put these in "gaps".

Return ONLY a valid JSON object with these fields:
  "coverage_score": integer 0-10 (10 = everything answerable from the documents is covered)
  "gaps": list of strings (answerable gaps only; empty if none)
  "interview_questions": list of strings
  "sufficient": boolean (true if no remaining answerable gap would change the recommendation)
  "rationale": one sentence explaining the coverage score

No markdown fences, no commentary — only the JSON object.`;

// ---------------------------------------------------------------------------
// Tool schema — screening spoke + refinement loop tools
// ---------------------------------------------------------------------------

const tools: Anthropic.Tool[] = [
  {
    name: "screening_agent",
    description:
      "Runs a specialist screening agent on one specific angle of candidate fit. " +
      "Each call receives the full job posting and resume. " +
      'In Phase 1, set "partition" to the partition agent name. ' +
      'In refinement, set "gap_index" to the index of the gap (from the latest ' +
      "evaluate_coverage result) this question fills.",
    input_schema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description: "The specific screening question to evaluate",
        },
        partition: {
          type: "string",
          description: "Phase 1 only: the partition agent name this call covers",
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function firstText(content: Anthropic.ContentBlock[]): string {
  const block = content.find(
    (b): b is Anthropic.TextBlock => b.type === "text",
  );
  return block?.text ?? "";
}

/** Strip markdown code fences if the model adds them anyway, then parse. */
function parseJson<T>(raw: string): T {
  let text = raw.trim();
  if (text.startsWith("```")) {
    text = text
      .split("\n")
      .filter((line) => !line.startsWith("```"))
      .join("\n")
      .trim();
  }
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    throw new Error(`Model returned invalid JSON:\n${raw}\n\n${String(err)}`);
  }
}

const normalize = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

function formatFindings(findings: Finding[]): string {
  return findings
    .map(
      (f, i) => `${i + 1}. [${f.source}]\n   Q: ${f.question}\n   A: ${f.answer}`,
    )
    .join("\n\n");
}

// ---------------------------------------------------------------------------
// Partition planner — generates non-overlapping screening scopes as JSON
// before any agent is invoked, so the coordinator can route without overlap.
// ---------------------------------------------------------------------------

const PARTITION_PLANNER_PROMPT = `You are a screening partition planner. Given a job posting and resume,
output a JSON array of non-overlapping screening partitions.

Each partition object must have:
  "agent"  — a short unique name  (e.g. "technical_depth_agent")
  "scope"  — an object with:
      "topic"   — one sentence describing what this partition evaluates
      "cover"   — list of specific aspects IN scope
      "exclude" — list of aspects explicitly OUT of scope (prevents overlap with other partitions)

Rules:
- Design partitions so that together they cover all relevant hiring questions
- No two partitions may share the same "cover" aspects
- Only include partitions that are genuinely needed for THIS candidate-role pair
- Return ONLY valid JSON — no markdown fences, no commentary`;

async function generatePartitions(
  client: Anthropic,
  jobPosting: string,
  resume: string,
): Promise<Partition[]> {
  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 4096,
    system: PARTITION_PLANNER_PROMPT,
    messages: [
      {
        role: "user",
        content:
          `Generate screening partitions for this application.\n\n` +
          `JOB POSTING:\n${jobPosting}\n\n` +
          `RESUME:\n${resume}`,
      },
    ],
  });
  return parseJson<Partition[]>(firstText(response.content));
}

async function callScreeningAgent(
  client: Anthropic,
  question: string,
  jobPosting: string,
  resume: string,
): Promise<string> {
  const r = await client.messages.create({
    model: MODEL,
    max_tokens: 2048,
    system: SCREENING_AGENT_PROMPT,
    messages: [
      {
        role: "user",
        content:
          `QUESTION: ${question}\n\n` +
          `JOB POSTING:\n${jobPosting}\n\n` +
          `RESUME:\n${resume}`,
      },
    ],
  });
  return firstText(r.content);
}

/** The evaluator sees the source documents plus the code-held findings — not a paraphrase. */
async function callEvaluationAgent(
  client: Anthropic,
  jobPosting: string,
  resume: string,
  findings: Finding[],
): Promise<CoverageEvaluation> {
  const r = await client.messages.create({
    model: MODEL,
    max_tokens: 4096,
    system: EVALUATION_AGENT_PROMPT,
    messages: [
      {
        role: "user",
        content:
          `JOB POSTING:\n${jobPosting}\n\n` +
          `RESUME:\n${resume}\n\n` +
          `SCREENING FINDINGS SO FAR:\n${formatFindings(findings)}`,
      },
    ],
  });
  const evaluation = parseJson<CoverageEvaluation>(firstText(r.content));
  evaluation.gaps ??= [];
  evaluation.interview_questions ??= [];
  return evaluation;
}

// ---------------------------------------------------------------------------
// Coordinator — refinement-loop agent
// ---------------------------------------------------------------------------

const DYNAMIC_COORDINATOR = `You are a job application screening coordinator with a refinement loop.

You will receive pre-planned screening PARTITIONS as JSON. Follow these phases:

PHASE 1 — Initial Screening:
Make exactly one screening_agent call per partition, with "partition" set to the partition's
agent name. Keep each question strictly within that partition's "cover" list and out of its
"exclude" list.

PHASE 2 — Evaluate Coverage:
Once every partition has been screened, call evaluate_coverage (no arguments — the system
passes it all findings, the job posting and the resume).

PHASE 3 — Refinement:
If the result has loop_status="continue", make one screening_agent call per gap, with
"gap_index" set to that gap's index in the latest "gaps" list. Then call evaluate_coverage again.

PHASE 4 — Submit Final:
When evaluate_coverage returns loop_status="stopped", call submit_final with your
HIRE / MAYBE / PASS recommendation and full justification. If stop_reason is not "sufficient",
say which gaps remain open. Always list the interview_questions to ask the candidate.
Do not call submit_final in the same turn as evaluate_coverage — wait for its result.

The system enforces these rules and rejects out-of-order calls.`;

const MAX_STEPS = 30;
/** Refinement rounds AFTER the initial evaluation. */
const MAX_REFINEMENTS = 3;

async function runCoordinator(
  client: Anthropic,
  coordinatorPrompt: string,
  label: string,
  jobPosting: string,
  resume: string,
): Promise<string[]> {
  const bar = "=".repeat(60);
  console.log(`\n${bar}\n${label}\n${bar}`);

  // Step 1 — generate non-overlapping partitions and show them to the human
  console.log("\nGenerating screening partitions...\n");
  const partitions = await generatePartitions(client, jobPosting, resume);
  console.log("Screening partitions:");
  console.log(JSON.stringify(partitions, null, 2));
  console.log();

  // Verify partitions have no duplicate cover items across agents
  const seen = new Set<string>();
  for (const p of partitions) {
    for (const item of p.scope?.cover ?? []) {
      const key = item.toLowerCase();
      if (seen.has(key)) {
        console.log(
          `  [OVERLAP WARNING] '${item}' appears in multiple partitions`,
        );
      }
      seen.add(key);
    }
  }

  console.log("Screening angles delegated:\n");

  // Step 2 — pass partitions to the coordinator so it routes correctly
  const partitionContext =
    "SCREENING PARTITIONS (non-overlapping scopes pre-planned for this candidate):\n" +
    JSON.stringify(partitions, null, 2);

  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content:
        `Please screen this application using the partitions below.\n\n` +
        `${partitionContext}\n\n` +
        `JOB POSTING:\n${jobPosting}\n\n` +
        `RESUME:\n${resume}`,
    },
  ];

  // ---- Loop state, owned by the code -------------------------------------
  const delegated: string[] = [];
  const findings: Finding[] = [];
  const partitionNames = new Set(partitions.map((p) => p.agent));
  const screenedPartitions = new Set<string>();
  let evaluations = 0; // 1 = initial evaluation, 2+ = after a refinement round
  let lastEval: CoverageEvaluation | null = null;
  let findingsAtLastEval = 0;
  const filledGaps = new Set<number>(); // gap indices filled since the last evaluation
  let stopReason: StopReason | null = null;

  const ok = (id: string, content: string): Anthropic.ToolResultBlockParam => ({
    type: "tool_result",
    tool_use_id: id,
    content,
  });
  const reject = (id: string, msg: string): Anthropic.ToolResultBlockParam => {
    console.log(`  [REJECTED] ${msg}`);
    return { type: "tool_result", tool_use_id: id, content: `REJECTED: ${msg}`, is_error: true };
  };

  async function handleScreening(
    id: string,
    input: Record<string, unknown>,
  ): Promise<Anthropic.ToolResultBlockParam> {
    if (stopReason) {
      return reject(id, `The loop has stopped (${stopReason}). Call submit_final.`);
    }
    const question = typeof input.question === "string" ? input.question.trim() : "";
    if (!question) return reject(id, "question is required.");

    let source: string;
    if (evaluations === 0) {
      // Phase 1 — exactly one call per known partition
      const name = input.partition;
      if (typeof name !== "string" || !partitionNames.has(name)) {
        return reject(
          id,
          `Phase 1 calls must set "partition" to one of: ${[...partitionNames].join(", ")}.`,
        );
      }
      if (screenedPartitions.has(name)) {
        return reject(id, `Partition '${name}' has already been screened.`);
      }
      screenedPartitions.add(name);
      source = `partition:${name}`;
    } else {
      // Refinement — every call must target an open gap from the latest evaluation
      const gaps = lastEval!.gaps;
      const idx = input.gap_index;
      if (typeof idx !== "number" || !Number.isInteger(idx) || idx < 0 || idx >= gaps.length) {
        return reject(
          id,
          `Refinement calls must set "gap_index" to 0-${gaps.length - 1} from the latest gaps list.`,
        );
      }
      if (filledGaps.has(idx)) {
        return reject(id, `Gap ${idx} has already been addressed this round.`);
      }
      filledGaps.add(idx);
      source = `gap:${gaps[idx]}`;
    }

    delegated.push(question);
    console.log(`  [${String(delegated.length).padStart(2)}] (${source}) ${question}`);
    const answer = await callScreeningAgent(client, question, jobPosting, resume);
    findings.push({ source, question, answer });
    return ok(id, answer);
  }

  async function handleEvaluate(id: string): Promise<Anthropic.ToolResultBlockParam> {
    if (stopReason) {
      return reject(id, `The loop has stopped (${stopReason}). Call submit_final.`);
    }
    if (evaluations === 0 && screenedPartitions.size < partitionNames.size) {
      const missing = [...partitionNames].filter((n) => !screenedPartitions.has(n));
      return reject(id, `Screen every partition first. Missing: ${missing.join(", ")}.`);
    }
    if (evaluations > 0 && findings.length === findingsAtLastEval) {
      return reject(id, "No new findings since the last evaluation — fill the gaps first.");
    }

    evaluations++;
    const round = evaluations - 1; // 0 = initial evaluation
    console.log(
      round === 0
        ? "\n[Evaluating coverage — initial]"
        : `\n[Evaluating coverage — refinement round ${round}/${MAX_REFINEMENTS}]`,
    );

    const evaluation = await callEvaluationAgent(client, jobPosting, resume, findings);
    const { gaps } = evaluation;
    console.log(`  Coverage score: ${evaluation.coverage_score ?? "?"}/10`);
    console.log(gaps.length ? `  Gaps: ${gaps.join(" | ")}` : "  No answerable gaps.");
    if (evaluation.interview_questions.length) {
      console.log(`  Interview-only: ${evaluation.interview_questions.join(" | ")}`);
    }
    console.log(`  Sufficient: ${evaluation.sufficient ?? false}`);

    // Stop conditions — decided by code, not by the model
    if (evaluation.sufficient) {
      stopReason = "sufficient";
    } else if (gaps.length === 0) {
      stopReason = "no_answerable_gaps";
    } else if (round >= MAX_REFINEMENTS) {
      stopReason = "limit_reached";
    } else if (lastEval) {
      const scoreImproved = evaluation.coverage_score > lastEval.coverage_score;
      const prevGaps = new Set(lastEval.gaps.map(normalize));
      const allGapsRepeated = gaps.every((g) => prevGaps.has(normalize(g)));
      if (!scoreImproved || allGapsRepeated) stopReason = "no_progress";
    }
    if (stopReason) console.log(`  [Loop stopped: ${stopReason}]`);

    lastEval = evaluation;
    findingsAtLastEval = findings.length;
    filledGaps.clear();

    // Report the real evaluation; never fake sufficient=true
    return ok(
      id,
      JSON.stringify({
        ...evaluation,
        refinement_round: round,
        max_refinements: MAX_REFINEMENTS,
        loop_status: stopReason ? "stopped" : "continue",
        stop_reason: stopReason,
        remaining_gaps: stopReason && stopReason !== "sufficient" ? gaps : [],
      }),
    );
  }

  function handleSubmit(
    id: string,
    input: Record<string, unknown>,
    turnHasEvaluate: boolean,
  ): Anthropic.ToolResultBlockParam | null {
    if (turnHasEvaluate) {
      return reject(id, "Wait for the evaluate_coverage result before calling submit_final.");
    }
    if (evaluations === 0) return reject(id, "Run evaluate_coverage first.");
    if (!stopReason) {
      return reject(id, "The loop is still open: fill the latest gaps, then re-evaluate.");
    }
    console.log(`\nFinal recommendation (${stopReason}):\n${String(input.recommendation)}`);
    return null; // accepted
  }

  // ---- Coordinator loop ---------------------------------------------------
  const order: Record<string, number> = {
    screening_agent: 0,
    evaluate_coverage: 1,
    submit_final: 2,
  };

  for (let step = 1; step <= MAX_STEPS; step++) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 8192,
      system: coordinatorPrompt,
      tools,
      messages,
    });

    if (response.stop_reason === "end_turn") {
      const final = firstText(response.content);
      if (final) console.log(`\nFinal recommendation:\n${final}`);
      return delegated;
    }

    if (response.stop_reason !== "tool_use") {
      // Without this guard, the same request would be re-sent until MAX_STEPS.
      console.log(
        `[WARN] Unexpected stop_reason '${response.stop_reason}' — stopping.`,
      );
      return delegated;
    }

    const toolUses = response.content.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
    );
    const turnHasEvaluate = toolUses.some((b) => b.name === "evaluate_coverage");

    // Run screening first, then evaluation, then submit — so an evaluation in the
    // same turn sees that turn's findings, and submit never runs ahead of it.
    const results = new Map<string, Anthropic.ToolResultBlockParam>();
    let submitted = false;
    for (const block of [...toolUses].sort(
      (a, b) => (order[a.name] ?? 3) - (order[b.name] ?? 3),
    )) {
      const input = (block.input ?? {}) as Record<string, unknown>;
      if (block.name === "screening_agent") {
        results.set(block.id, await handleScreening(block.id, input));
      } else if (block.name === "evaluate_coverage") {
        results.set(block.id, await handleEvaluate(block.id));
      } else if (block.name === "submit_final") {
        const rejection = handleSubmit(block.id, input, turnHasEvaluate);
        if (rejection) {
          results.set(block.id, rejection);
        } else {
          results.set(block.id, ok(block.id, "Recommendation submitted."));
          submitted = true;
        }
      } else {
        results.set(block.id, reject(block.id, `Unknown tool '${block.name}'.`));
      }
    }

    messages.push(
      { role: "assistant", content: response.content },
      { role: "user", content: toolUses.map((b) => results.get(b.id)!) },
    );

    if (submitted) return delegated;
  }

  console.log("[WARN] Reached step limit.");
  return delegated;
}

// ---------------------------------------------------------------------------
// Coverage report
// ---------------------------------------------------------------------------

function coverageReport(dynamicQs: string[]): void {
  const dimensions: [string, string[]][] = [
    [
      "technical / hard skill match",
      ["skill", "python", "fastapi", "keyword", "technical", "required"],
    ],
    [
      "experience depth / seniority",
      ["depth", "senior", "experience", "level", "scope"],
    ],
    [
      "red flags / disqualifiers",
      ["red flag", "gap", "hop", "disqualif", "concern"],
    ],
    [
      "soft skills / working style",
      ["soft", "communication", "collaborat", "working style", "team"],
    ],
    [
      "growth trajectory / potential",
      ["growth", "trajectory", "potential", "progression", "future"],
    ],
    [
      "nice-to-haves and trade-offs",
      ["nice", "trade", "bonus", "preferred", "optional"],
    ],
    [
      "compensation / motivation / timeline",
      ["compens", "salary", "motiv", "timeline", "expectation"],
    ],
    [
      "team / culture fit",
      ["culture", "team fit", "team dynamic", "environment"],
    ],
    [
      "compensating strengths for gaps",
      ["compensat", "offset", "strength", "despite", "mitigat"],
    ],
  ];

  const combined = dynamicQs.join(" ").toLowerCase();
  const covered = (keywords: string[]) =>
    keywords.some((kw) => combined.includes(kw));

  const bar = "=".repeat(60);
  console.log(`\n${bar}\nCOVERAGE REPORT\n${bar}`);
  console.log(
    `\n  Dynamic coordinator:  ${String(dynamicQs.length).padStart(2)} screening angles\n`,
  );
  console.log(`  ${"Dimension".padEnd(45)} ${"Covered".padStart(8)}`);
  console.log(`  ${"-".repeat(45)} ${"-".repeat(8)}`);

  for (const [label, keywords] of dimensions) {
    console.log(
      `  ${label.padEnd(45)} ${(covered(keywords) ? "✓" : "✗").padStart(8)}`,
    );
  }

  console.log();
  console.log(
    "  DYNAMIC  — reads the candidate first, routes to relevant checks only.",
  );
  console.log(
    "             Skips obvious angles, adds targeted ones where risk exists.",
  );
  console.log("             Fewer agents, sharper signal.");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const jobPosting = `Title: Senior Python Backend Engineer
Required skills: Python (5+ years), FastAPI, PostgreSQL, Docker, REST API design
Nice to have: Kubernetes, Redis, system design at scale
Seniority: Senior (5-8 years total experience)`;

  const resume = `Alex Chen — 7 years experience
2022–present: Backend Engineer, FinTech startup — Python, FastAPI, PostgreSQL, Docker
2019–2022:    Software Engineer, SaaS company — Python, Django, MySQL
2017–2019:    Junior Developer, agency — PHP, JavaScript
Skills: Python, FastAPI, PostgreSQL, Docker, REST APIs, some Redis
Education: B.Sc. Computer Science`;

  console.log(
    "Candidate: Alex Chen — Senior Python Backend Engineer application\n",
  );
  console.log(
    "The coordinator reads the candidate first, then routes to relevant checks only.",
  );

  const dynamicQs = await runCoordinator(
    client,
    DYNAMIC_COORDINATOR,
    "DYNAMIC COORDINATOR  (reads candidate first, routes to relevant checks only)",
    jobPosting,
    resume,
  );

  coverageReport(dynamicQs);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
