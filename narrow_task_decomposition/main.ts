import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

const MODEL = "claude-haiku-5-5";

const COORDINATOR_PROMPT = `You are a job application screening coordinator.
Your job is to orchestrate three independent screening agents and then aggregate their results.

Steps:
1. Run all three screening agents - keyword_scanner, deep_evaluator, red_flag_detector - passing the full job_posting and resume text to each
2. Once all three have returned results, pass their outputs to run_score_aggregator.
3. Present the final recommendation to the user.

You may run the three screening agents in any order. Do not skip any of them
`;

// Narrow decomposition: each subagent answers one question and is told to leave
// everything else to the others. None of them can see the others' results.
const subagents: Record<string, string> = {
  keyword_scanner: `You check one thing: which skills and keywords required by the job posting literally appear in the resume.
List each one as FOUND or MISSING, then end with "Score: N/100".
Do not judge how strong the experience is or whether the resume looks suspicious. Other agents cover that.`,
  deep_evaluator: `You check one thing: how well the candidate's actual experience fits the responsibilities of the role.
Look at depth, seniority and relevance of the work described. Write at most 3 bullet points, then end with "Score: N/100".
Do not count keywords or look for red flags. Other agents cover that.`,
  red_flag_detector: `You check one thing: warning signs in the resume, such as employment gaps, very short tenures, or claims that contradict each other.
List each one with the evidence from the resume, or say "No red flags". End with "Risk: LOW", "Risk: MEDIUM" or "Risk: HIGH".
Do not assess whether the candidate fits the role. Other agents cover that.`,
  run_score_aggregator: `You combine the results of three independent screening agents into one hiring recommendation.
Use only the results you are given. Do not re-evaluate the candidate yourself.
Reply with ADVANCE, HOLD or REJECT, an overall score out of 100, and 2 sentences of reasoning.
If the agents disagree, say so.`,
};

async function runSubagent(agent: string, input: Record<string, string>): Promise<string> {
  const system = subagents[agent];
  if (!system) return `Unknown agent: ${agent}`;

  const content = Object.entries(input)
    .map(([name, value]) => `<${name}>\n${value}\n</${name}>`)
    .join("\n");

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 2048,
    system,
    messages: [{ role: "user", content }],
  });

  const text = response.content.find((block) => block.type === "text");
  return text?.text ?? "";
}

const screeningInput: Anthropic.Tool.InputSchema = {
  type: "object",
  properties: {
    job_posting: { type: "string", description: "The full job posting text" },
    resume: { type: "string", description: "The full resume text" },
  },
  required: ["job_posting", "resume"],
};

const tools: Anthropic.Tool[] = [
  {
    name: "keyword_scanner",
    description: "Checks which required skills and keywords from the job posting appear in the resume.",
    input_schema: screeningInput,
  },
  {
    name: "deep_evaluator",
    description: "Evaluates how well the candidate's experience fits the responsibilities of the role.",
    input_schema: screeningInput,
  },
  {
    name: "red_flag_detector",
    description: "Looks for warning signs in the resume, such as gaps, short tenures, or contradictions.",
    input_schema: screeningInput,
  },
  {
    name: "run_score_aggregator",
    description: "Combines the outputs of all three screening agents into a final recommendation.",
    input_schema: {
      type: "object",
      properties: {
        keyword_scanner_result: { type: "string", description: "The full output of keyword_scanner" },
        deep_evaluator_result: { type: "string", description: "The full output of deep_evaluator" },
        red_flag_detector_result: { type: "string", description: "The full output of red_flag_detector" },
      },
      required: ["keyword_scanner_result", "deep_evaluator_result", "red_flag_detector_result"],
    },
  },
];

const jobPosting = `Senior Backend Engineer
Build and operate the payment APIs behind our checkout. You will design services, own them in production and mentor two junior engineers.
Required: TypeScript, Node.js, PostgreSQL, AWS, 5+ years of backend experience.
Nice to have: Kafka, payments domain experience.`;

// Fictional candidate: strong keyword match, but the dates are worth a second look.
const resume = `Sam Rivera - Software Engineer
Skills: TypeScript, Node.js, PostgreSQL, AWS, Docker, Kafka

Experience
- Backend Engineer, Northwind Pay (Mar 2024 - Sep 2024): built refund APIs in Node.js and PostgreSQL.
- Software Engineer, Brightly (Jan 2023 - Oct 2023): maintained internal admin tools in TypeScript.
- Junior Developer, Acme Web (Jun 2020 - Jul 2021): built marketing sites in WordPress.

Summary: 8 years of experience leading backend teams at scale.`;

const userMessage = `Please screen this application.

<job_posting>
${jobPosting}
</job_posting>

<resume>
${resume}
</resume>`;

const messages: Anthropic.MessageParam[] = [{ role: "user", content: userMessage }];

// Coordinator loop: keep going until the coordinator decides it is finished.
for (let turn = 1; turn <= 10; turn++) {
  const response = await client.messages.create({
    model: MODEL,
    // Higher than the subagents: the coordinator copies the full texts into every tool call.
    max_tokens: 4096,
    system: COORDINATOR_PROMPT,
    tools,
    messages,
  });

  console.log(`\n[turn ${turn}] stop_reason: ${response.stop_reason}`);
  messages.push({ role: "assistant", content: response.content });

  if (response.stop_reason !== "tool_use") {
    const text = response.content.find((block) => block.type === "text");
    console.log(`\nCoordinator: ${text?.text}`);
    break;
  }

  // The agents are independent, so any calls made in the same turn can run in parallel.
  const toolUses = response.content.filter((block) => block.type === "tool_use");
  const toolResults = await Promise.all(
    toolUses.map(async (block): Promise<Anthropic.ToolResultBlockParam> => {
      const input = block.input as Record<string, string>;
      console.log(`Coordinator -> ${block.name} (${Object.keys(input).join(", ")})`);
      const result = await runSubagent(block.name, input);
      console.log(`\n${block.name} -> Coordinator:\n${result}`);
      return { type: "tool_result", tool_use_id: block.id, content: result };
    }),
  );
  messages.push({ role: "user", content: toolResults });
}
