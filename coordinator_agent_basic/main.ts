import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

const MODEL = "claude-haiku-5-5";

// Each subagent is just a model call with its own system prompt.
// It only sees the task the coordinator writes for it, never the full conversation.
const subagents: Record<string, string> = {
  researcher: `You are a researcher. Answer the task with a short list of key facts.
No intro, no conclusion, just the facts.`,
  writer: `You are a writer. Turn the notes you are given into one friendly, clear paragraph.`,
};

async function runSubagent(agent: string, task: string): Promise<string> {
  const system = subagents[agent];
  if (!system) return `Unknown agent: ${agent}`;

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    system,
    messages: [{ role: "user", content: task }],
  });

  const text = response.content.find((block) => block.type === "text");
  return text?.text ?? "";
}

// The coordinator has one capability: hand work to a subagent.
const tools: Anthropic.Tool[] = [
  {
    name: "delegate",
    description:
      "Sends a self-contained task to a subagent and returns its answer. The subagent cannot see this conversation, so include everything it needs in the task.",
    input_schema: {
      type: "object",
      properties: {
        agent: {
          type: "string",
          enum: Object.keys(subagents),
          description: "researcher gathers facts, writer turns notes into polished text",
        },
        task: { type: "string", description: "The full instructions for the subagent" },
      },
      required: ["agent", "task"],
    },
  },
];

const system = `You are a coordinator. You do not do the work yourself.
Break the user's request into steps and delegate each step to a subagent:
- Use researcher to gather facts.
- Use writer to turn those facts into the final answer, passing it the research notes.
When the writer is done, return its text to the user unchanged.`;

const userMessage = "Explain to a 10 year old why the sky is blue.";

const messages: Anthropic.MessageParam[] = [{ role: "user", content: userMessage }];

// Coordinator loop: keep going until the coordinator decides it is finished.
for (let turn = 1; turn <= 10; turn++) {
  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    system,
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
  console.log(response.content)
  const toolResults: Anthropic.ToolResultBlockParam[] = [];
  for (const block of response.content) {
    if (block.type !== "tool_use") continue;
    const { agent, task } = block.input as Record<string, string>;
    console.log(`Coordinator -> ${agent}: ${task}`);
    const result = await runSubagent(agent, task);
    console.log(`${agent} -> Coordinator: ${result}`);
    toolResults.push({ type: "tool_result", tool_use_id: block.id, content: result });
  }
  messages.push({ role: "user", content: toolResults });
}
