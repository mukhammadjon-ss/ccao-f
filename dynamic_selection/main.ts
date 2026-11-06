import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

const MODEL = "claude-haiku-5-5";

// Dynamic selection: the prompt gives the coordinator a goal and selection criteria,
// not a fixed list of steps. Each application can take a different path.
const COORDINATOR_PROMPT = `You are a job application screening coordinator.
Decide whether to ADVANCE, HOLD or REJECT the application using as few specialist agents as you can.

Read the application first, then pick the agent most likely to settle the decision.
After each result, either pick another agent or stop.

- Only call an agent if its answer could change your recommendation.
- Stop as soon as the recommendation is clear.
- Call each agent at most once, and tell it the specific question to answer for this application.
- Base your recommendation on the agents' findings. Call at least one agent.

When you are done, reply with ADVANCE, HOLD or REJECT on the first line, then 2 sentences of reasoning.`;

// Each subagent still answers one narrow question. What changed is who decides when it runs.
const subagents: Record<string, string> = {
  keyword_scanner: `You check one thing: which skills and keywords required by the job posting literally appear in the resume.
List each one as FOUND or MISSING, then end with "Score: N/100".`,
  deep_evaluator: `You check one thing: how well the work described in the candidate's listed jobs fits the responsibilities of the role.
Look at depth, seniority and relevance. Ignore dates, the summary and any personal or open source projects, even if the focus mentions them.
Write at most 3 bullet points, then end with "Score: N/100".`,
  red_flag_detector: `You check one thing: warning signs in the resume, such as very short tenures or claims that contradict each other.
Do not assess whether the candidate fits the role.
List each one with the evidence from the resume, or say "No red flags". End with "Risk: LOW", "Risk: MEDIUM" or "Risk: HIGH".`,
  project_reviewer: `You check one thing: what the projects described in the resume show about the candidate's ability to do this job.
Ignore job titles and employment history.
Write at most 3 bullet points, then end with "Score: N/100".`,
};

async function runSubagent(
  agent: string,
  focus: string,
  application: string,
): Promise<string> {
  const system = subagents[agent];
  if (!system) return `Unknown agent: ${agent}`;

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 2048,
    system,
    messages: [
      {
        role: "user",
        content: `${application}\n\n<focus>\n${focus}\n</focus>`,
      },
    ],
  });

  const text = response.content.find((block) => block.type === "text");
  return text?.text ?? "";
}

// The code passes the application to the agent, so the coordinator only writes the focus.
const agentInput: Anthropic.Tool.InputSchema = {
  type: "object",
  properties: {
    focus: {
      type: "string",
      description: "The specific question this agent should answer for this application",
    },
  },
  required: ["focus"],
};

// The descriptions are the selection criteria: they say when each agent is worth calling.
const tools: Anthropic.Tool[] = [
  {
    name: "keyword_scanner",
    description:
      "Checks which required skills from the job posting appear in the resume. Use when it is unclear whether the candidate has the required skills at all.",
    input_schema: agentInput,
  },
  {
    name: "deep_evaluator",
    description:
      "Evaluates how well the work in the candidate's listed jobs fits the role. It ignores dates, the summary and projects, so it cannot check whether claims are true. Use when the candidate looks plausible and the depth or seniority of their jobs needs judging.",
    input_schema: agentInput,
  },
  {
    name: "red_flag_detector",
    description:
      "Looks for warning signs such as very short tenures or contradictory claims. Use when dates or claims in the resume look inconsistent.",
    input_schema: agentInput,
  },
  {
    name: "project_reviewer",
    description:
      "Assesses the projects described in the resume, ignoring employment. Use when the resume describes substantial projects and job titles alone do not show what the candidate can do.",
    input_schema: agentInput,
  },
];

const jobPosting = `Senior Backend Engineer
Build and operate the payment APIs behind our checkout. You will design services, own them in production and mentor two junior engineers.
Required: TypeScript, Node.js, PostgreSQL, AWS, 5+ years of backend experience.
Nice to have: Kafka, payments domain experience.`;

// Fictional candidates, each written to need a different set of agents.
const candidates = [
  {
    // Obvious mismatch: one agent should be enough.
    name: "Jordan Lee",
    resume: `Jordan Lee - Graphic Designer
Skills: Photoshop, Illustrator, Figma

Experience
- Senior Designer, Inkwell Studio (Feb 2017 - present): brand identity and packaging design.

Summary: 9 years of experience in print and brand design.`,
  },
  {
    // Looks like a strong fit, but the dates contradict the summary.
    name: "Sam Rivera",
    resume: `Sam Rivera - Senior Backend Engineer
Skills: TypeScript, Node.js, PostgreSQL, AWS, Docker, Kafka

Experience
- Senior Backend Engineer, Northwind Pay (Mar 2024 - Sep 2024): designed and owned refund APIs in TypeScript, Node.js and PostgreSQL on AWS, and mentored two engineers.
- Backend Engineer, Brightly (Jan 2023 - Oct 2023): built billing services in Node.js and Kafka and ran them in production.
- Backend Engineer, Corvid Labs (Jun 2022 - Nov 2022): built REST APIs in TypeScript and PostgreSQL.

Summary: 8 years of experience leading backend teams at scale.`,
  },
  {
    // Job titles undersell the candidate: the projects are the real evidence.
    name: "Alex Morgan",
    resume: `Alex Morgan - Data Analyst
Skills: SQL, Python, TypeScript, Node.js, PostgreSQL

Experience
- Data Analyst, Fernway Retail (Apr 2018 - present): reporting and dashboards. Since 2021, also on-call owner of the internal billing API.

Projects
- LedgerLite: open source double-entry payments ledger in TypeScript and PostgreSQL, deployed on AWS, handling 2M transactions a month for three companies. Sole maintainer since 2020, mentoring four regular contributors.
- Webhook relay in Node.js with idempotent retries, used in production by LedgerLite customers.`,
  },
];

async function screen(name: string, resume: string) {
  console.log(`\n=== ${name} ===`);

  const application = `<job_posting>
${jobPosting}
</job_posting>

<resume>
${resume}
</resume>`;

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: `Please screen this application.\n\n${application}` },
  ];
  const agentsUsed: string[] = [];

  // Coordinator loop: keep going until the coordinator decides it is finished.
  for (let turn = 1; turn <= 10; turn++) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: COORDINATOR_PROMPT,
      tools,
      // One agent per turn, so each choice can depend on the previous result.
      tool_choice: { type: "auto", disable_parallel_tool_use: true },
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    const toolUse = response.content.find((block) => block.type === "tool_use");
    if (response.stop_reason !== "tool_use" || !toolUse) {
      const text = response.content.find((block) => block.type === "text");
      console.log(`\nCoordinator (${response.stop_reason}): ${text?.text}`);
      return { candidate: name, agents: agentsUsed.join(", "), decision: text?.text.split("\n")[0] };
    }

    const { focus } = toolUse.input as { focus: string };
    console.log(`\nCoordinator -> ${toolUse.name}: ${focus}`);
    const result = await runSubagent(toolUse.name, focus, application);
    console.log(`${toolUse.name} -> Coordinator:\n${result}`);
    agentsUsed.push(toolUse.name);

    messages.push({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: toolUse.id, content: result }],
    });
  }

  return { candidate: name, agents: agentsUsed.join(", "), decision: "no decision after 10 turns" };
}

const results = [];
for (const { name, resume } of candidates) {
  results.push(await screen(name, resume));
}

// Same coordinator, same prompt: compare which agents each application needed.
console.log("\n=== Agents selected per application ===");
console.table(results);
