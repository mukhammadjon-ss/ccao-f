import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

// The code only supplies capabilities. There is no switch statement:
// the model decides which tools to call, in what order, and when it is done.
const tools: Anthropic.Tool[] = [
  {
    name: "magic_eyeball",
    description: "Answers a yes or no fortune telling question about the future.",
    input_schema: {
      type: "object",
      properties: {
        question: { type: "string", description: "The yes/no question to ask" },
      },
      required: ["question"],
    },
  },
  {
    name: "get_weather",
    description: "Gets the current weather for a city.",
    input_schema: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name, e.g. Tashkent" },
      },
      required: ["city"],
    },
  },
];

// Demonstration only: this is where the decision tree used to live.
// Modern Claude infers all of this from the tool descriptions alone,
// but spelling it out shows that the routing rules moved from code into the prompt.
const system = `You are a helpful assistant with access to tools.
Decide for yourself which tools, if any, are needed to answer the user:
- If the user asks a yes/no question about the future, call magic_eyeball.
- If the user asks about the weather, call get_weather for each city mentioned.
- If the user just says hello or makes small talk, reply directly without tools.
- If the request needs several tools, call all of them before answering.
- If nothing fits, say what you can help with.
When you have what you need, answer the user in one short paragraph.`;

// Fake implementations. The code executes tools but never chooses them.
function runTool(name: string, input: Record<string, string>): string {
  switch (name) {
    case "magic_eyeball": {
      const answers = ["Yes", "No", "Ask again later"];
      return answers[Math.floor(Math.random() * answers.length)];
    }
    case "get_weather":
      return JSON.stringify({ city: input.city, tempC: 18, conditions: "light rain" });
    default:
      return `Unknown tool: ${name}`;
  }
}

const userMessage =
  "Hey Claude, will I be a millionaire in 2027? Also, should I bring an umbrella in Tashkent today?";

const messages: Anthropic.MessageParam[] = [{ role: "user", content: userMessage }];

// Agent loop: keep going until the model decides it is finished.
for (let turn = 1; turn <= 10; turn++) {
  const response = await client.messages.create({
    model: "claude-haiku-5-5",
    max_tokens: 1024,
    system,
    tools,
    messages,
  });

  console.log(`\n[turn ${turn}] stop_reason: ${response.stop_reason}`);
  messages.push({ role: "assistant", content: response.content });

  if (response.stop_reason !== "tool_use") {
    const text = response.content.find((block) => block.type === "text");
    console.log(`\nClaude: ${text?.text}`);
    break;
  }

  const toolResults: Anthropic.ToolResultBlockParam[] = [];
  for (const block of response.content) {
    if (block.type !== "tool_use") continue;
    const result = runTool(block.name, block.input as Record<string, string>);
    console.log(`Claude decided to call ${block.name}(${JSON.stringify(block.input)}) -> ${result}`);
    toolResults.push({ type: "tool_result", tool_use_id: block.id, content: result });
  }
  messages.push({ role: "user", content: toolResults });
}
