import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

const tools: Anthropic.Tool[] = [
  {
    name: "magic_eyeball",
    description:
      "When the user asks a yes or no fortune telling question call this function",
    input_schema: {
      type: "object",
      properties: {
        question: { type: "string" },
      },
      required: ["question"],
    },
  },
];

const messages: Anthropic.MessageParam[] = [
  { role: "user", content: "Hey Claude, will I be a millionaire in 2027 ?" },
];

const response = await client.messages.create({
  max_tokens: 1024,
  messages,
  model: "claude-opus-5-5",
  tools,
  tool_choice: { type: "auto" },
});

const toolUse = response.content.find((block) => block.type === "tool_use")!;
console.log(
  `Claude called ${toolUse.name} with ${JSON.stringify(toolUse.input)}`,
);

console.log(response)

const reply = "Yes";

messages.push(
  { role: "assistant", content: response.content },
  {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: toolUse.id, content: reply }],
  },
);

const followup = await client.messages.create({
  model: "claude-opus-5-5",
  max_tokens: 1024,
  tools,
  tool_choice: { type: "auto", disable_parallel_tool_use: true },
  messages,
});

// Claude uses the result to answer the original question.
const finalText = followup.content.find((block) => block.type === "text")!;
console.log(followup);