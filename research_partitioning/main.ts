import Anthropic from "@anthropic-ai/sdk";

process.loadEnvFile();
// ---------------------------------------------------------------------------
// Use case: Job Application Screener — hub-and-spoke multi-agent pipeline
// ---------------------------------------------------------------------------

const MODEL = "claude-haiku-4-5-20251001";

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
  gaps: string[];
  sufficient: boolean;
  rationale: string;
}

// ---------------------------------------------------------------------------
// Spoke system prompts
// ---------------------------------------------------------------------------

const SCREENING_AGENT_PROMPT = `You are a specialist hiring analyst. You will be given a specific
screening question about a candidate along with the job posting and resume.

Answer the question in 2-3 focused sentences. Be concrete and specific to this candidate.`;

const EVALUATION_AGENT_PROMPT = `You are a screening coverage evaluator. Review the screening
findings collected so far and identify gaps.

Return ONLY a valid JSON object with these fields:
  "coverage_score": integer 0-10 (10 = complete coverage of all important hiring dimensions)
  "gaps": list of strings describing specific missing dimensions or unanswered questions
  "sufficient": boolean (true if a confident final recommendation can be made right now)
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
      "Call once per screening dimension with a focused question. " +
      "Each call receives the full job posting and resume.",
    input_schema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description: "The specific screening question to evaluate",
        },
      },
      required: ["question"],
    },
  },
  {
    name: "evaluate_coverage",
    description:
      "Evaluate the current screening findings for completeness and identify gaps. " +
      "Call after the initial partition agents have all reported, and again after " +
      "each refinement round. Returns a coverage score, list of gaps, and whether " +
      "coverage is now sufficient to make a final recommendation.",
    input_schema: {
      type: "object",
      properties: {
        current_findings: {
          type: "string",
          description: "Summary of all screening results collected so far",
        },
      },
      required: ["current_findings"],
    },
  },
  {
    name: "submit_final",
    description:
      "Submit the final HIRE / MAYBE / PASS recommendation. " +
      "Only call this when evaluate_coverage has confirmed sufficient coverage, " +
      "or when the refinement iteration limit has been reached.",
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
    max_tokens: 1024,
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
    max_tokens: 200,
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

async function callEvaluationAgent(
  client: Anthropic,
  currentFindings: string,
): Promise<CoverageEvaluation> {
  const r = await client.messages.create({
    model: MODEL,
    max_tokens: 400,
    system: EVALUATION_AGENT_PROMPT,
    messages: [
      {
        role: "user",
        content: `Evaluate coverage of these screening findings:\n\n${currentFindings}`,
      },
    ],
  });
  return parseJson<CoverageEvaluation>(firstText(r.content));
}

// ---------------------------------------------------------------------------
// Coordinator — refinement-loop agent
// ---------------------------------------------------------------------------

const DYNAMIC_COORDINATOR = `You are a job application screening coordinator with a refinement loop.

You will receive pre-planned screening PARTITIONS as JSON. Follow these phases:

PHASE 1 — Initial Screening:
Invoke exactly one screening_agent call per partition. Formulate each question so it stays
strictly within that partition's "cover" list and avoids its "exclude" list.

PHASE 2 — Evaluate Coverage:
After all initial partition agents have reported, call evaluate_coverage with a plain-text
summary of all findings collected so far. The evaluator will return a coverage score,
identified gaps, and whether coverage is sufficient.

PHASE 3 — Refinement (max 3 iterations):
If evaluate_coverage returns sufficient=false and gaps exist:
  - Invoke screening_agent calls to fill only the identified gaps (one call per gap).
  - Then call evaluate_coverage again with the updated findings summary.
  - Repeat until sufficient=true or 3 refinement iterations are exhausted.

PHASE 4 — Submit Final:
Call submit_final with your HIRE / MAYBE / PASS recommendation and full justification
once evaluate_coverage confirms sufficient=true, or once you have reached the iteration limit.

Do NOT call submit_final before running evaluate_coverage at least once.`;

const MAX_STEPS = 30;
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

  const delegated: string[] = [];
  let refinementIter = 0;

  for (let step = 1; step <= MAX_STEPS; step++) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
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

    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    let submitted = false;

    for (const block of response.content) {
      if (block.type !== "tool_use") continue;
      const input = block.input as Record<string, string>;

      if (block.name === "screening_agent") {
        const question = input.question;
        delegated.push(question);
        console.log(`  [${String(delegated.length).padStart(2)}] ${question}`);
        const result = await callScreeningAgent(
          client,
          question,
          jobPosting,
          resume,
        );
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      } else if (block.name === "evaluate_coverage") {
        refinementIter++;
        console.log(`\n[Evaluating coverage — iteration ${refinementIter}]`);
        const evaluation = await callEvaluationAgent(
          client,
          input.current_findings,
        );
        const gaps = evaluation.gaps ?? [];
        console.log(`  Coverage score: ${evaluation.coverage_score ?? "?"}/10`);
        console.log(
          gaps.length ? `  Gaps: ${gaps.join(", ")}` : "  No gaps identified.",
        );
        console.log(`  Sufficient: ${evaluation.sufficient ?? false}`);

        // If we've hit the refinement cap, force sufficient=true
        if (refinementIter >= MAX_REFINEMENTS && !evaluation.sufficient) {
          console.log("  [Refinement limit reached — proceeding to final]");
          evaluation.sufficient = true;
        }

        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: JSON.stringify(evaluation),
        });
      } else if (block.name === "submit_final") {
        console.log(`\nFinal recommendation:\n${input.recommendation}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: "Recommendation submitted.",
        });
        submitted = true;
      }
    }

    messages.push(
      { role: "assistant", content: response.content },
      { role: "user", content: toolResults },
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
