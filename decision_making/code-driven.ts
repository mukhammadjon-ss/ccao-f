import { Anthropic } from "@anthropic-ai/sdk";

process.loadEnvFile();

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

// The LLM only classifies. Every decision after that is hardcoded below.
const CATEGORIES = ["fortune", "weather", "greeting", "other"] as const;
type Category = (typeof CATEGORIES)[number];

async function classify(userMessage: string): Promise<Category> {
  const response = await client.messages.create({
    model: "claude-opus-5-5",
    max_tokens: 10,
    system: `Classify the user's message into exactly one of these categories: ${CATEGORIES.join(", ")}.
- fortune: a yes/no fortune telling question about the future
- weather: a question about the weather
- greeting: a simple hello or small talk
- other: anything else
Reply with only the category name, lowercase, nothing else.`,
    messages: [{ role: "user", content: userMessage }],
  });

  const text = response.content.find((block) => block.type === "text");
  const label = text?.text.trim().toLowerCase();

  // Code decides what to do with unexpected output, not the model.
  return CATEGORIES.includes(label as Category) ? (label as Category) : "other";
}

const userMessage = "Hey Claude, will I be a millionaire in 2027 ?";

const category = await classify(userMessage);
console.log(`Claude classified the message as: ${category}`);

// Pre-configured decision tree: each branch is fixed in code.
let reply: string;
switch (category) {
  case "fortune": {
    const answers = ["Yes", "No", "Ask again later"];
    reply = `The magic eyeball says: ${answers[Math.floor(Math.random() * answers.length)]}`;
    break;
  }
  case "weather":
    reply = "Sorry, I can't check the weather from here. Try a weather app.";
    break;
  case "greeting":
    reply = "Hello! Ask me a yes or no question about your future.";
    break;
  case "other":
    reply = "I only answer fortune telling questions.";
    break;
}

console.log(reply);
