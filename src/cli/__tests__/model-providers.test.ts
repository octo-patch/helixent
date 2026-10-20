import Anthropic from "@anthropic-ai/sdk";
import { describe, expect, test } from "bun:test";
import { OpenAI } from "openai";

import { AnthropicModelProvider } from "@/community/anthropic";
import { OpenAIModelProvider } from "@/community/openai";
import { type Message, Model } from "@/foundation";

import { helixentConfigSchema } from "../config";
import { getDefaultModelOptions, MODEL_PROVIDERS } from "../model-providers";

const MINIMAX_PROVIDERS = [
  { id: "minimax_cn", baseURL: "https://api.minimaxi.com/v1", providerType: "openai" },
  { id: "minimax_global", baseURL: "https://api.minimax.io/v1", providerType: "openai" },
  { id: "minimax_anthropic_cn", baseURL: "https://api.minimaxi.com/anthropic", providerType: "anthropic" },
  { id: "minimax_anthropic_global", baseURL: "https://api.minimax.io/anthropic", providerType: "anthropic" },
] as const;

const HISTORY: Message[] = [
  {
    role: "user",
    content: [
      { type: "text", text: "Inspect this image" },
      { type: "image_url", image_url: { url: "https://example.com/image.png" } },
    ],
  },
  {
    role: "assistant",
    content: [
      { type: "thinking", thinking: "Prior reasoning" },
      { type: "tool_use", id: "call_test", name: "inspect", input: { path: "image.png" } },
    ],
  },
  { role: "tool", content: [{ type: "tool_result", tool_use_id: "call_test", content: "Image inspected" }] },
  { role: "user", content: [{ type: "text", text: "Continue" }] },
];

const REQUEST_CASES: { name: string; model?: string; options?: Record<string, unknown> }[] = [
  { name: "wizard default" },
  { name: "saved Flash", model: "MiniMax-M3.1-Flash" },
  { name: "saved M3", model: "MiniMax-M3" },
  { name: "saved legacy", model: "MiniMax-M2.7" },
  { name: "custom model", model: "custom-model" },
  { name: "explicit M3 disabled", model: "MiniMax-M3", options: { thinking: { type: "disabled" } } },
  ...["low", "medium", "high", "xhigh", "max"].map((effort) => ({
    name: `Flash effort ${effort}`,
    model: "MiniMax-M3.1-Flash",
    options: { effort },
  })),
];

describe("MiniMax providers", () => {
  test("suggest Flash on the direct endpoints", () => {
    for (const expected of MINIMAX_PROVIDERS) {
      expect(MODEL_PROVIDERS.find(({ id }) => id === expected.id)).toMatchObject({
        ...expected,
        defaultModelName: "MiniMax-M3.1-Flash",
      });
    }
  });

  test("preserve protocol-specific M3, legacy and custom defaults", () => {
    for (const protocol of ["openai", "anthropic"] as const) {
      expect(getDefaultModelOptions("MiniMax-M3.1-Flash", protocol)).toEqual({
        max_tokens: 16 * 1024,
        thinking: { type: "adaptive" },
      });
      expect(getDefaultModelOptions("MiniMax-M2.7", protocol)).toEqual({ max_tokens: 16 * 1024 });
      expect(getDefaultModelOptions("custom-model", protocol)).toEqual({
        max_tokens: 16 * 1024,
        thinking: { type: "enabled" },
      });
    }
    expect(getDefaultModelOptions("MiniMax-M3", "openai")).toEqual({
      max_tokens: 16 * 1024,
      thinking: { type: "adaptive" },
    });
    expect(getDefaultModelOptions("MiniMax-M3", "anthropic")).toEqual({ max_tokens: 16 * 1024 });
  });

  for (const expected of MINIMAX_PROVIDERS) {
    for (const requestCase of REQUEST_CASES) {
      for (const stream of [false, true]) {
        test(`${expected.id} serializes ${requestCase.name} (stream=${stream})`, async () => {
          const config = MODEL_PROVIDERS.find(({ id }) => id === expected.id)!;
          const selectedModel = requestCase.model ?? config.defaultModelName!;
          const savedConfig = helixentConfigSchema.parse({
            models: [{ name: selectedModel, baseURL: config.baseURL, APIKey: "test-api-key", provider: config.providerType }],
            defaultModel: selectedModel,
          });
          const entry = savedConfig.models.find(({ name }) => name === savedConfig.defaultModel)!;
          const modelName = entry.name;
          expect(modelName).toBe(selectedModel);
          const anthropic = entry.provider === "anthropic";
          let requestURL = "";
          let body: Record<string, unknown> = {};
          const fetchRequest = async (input: RequestInfo | URL, init?: RequestInit) => {
            const request = new Request(input, init);
            requestURL = request.url;
            body = await request.json();
            if (stream) {
              return new Response(
                anthropic ? 'event: message_stop\ndata: {"type":"message_stop"}\n\n' : "data: [DONE]\n\n",
                { headers: { "content-type": "text/event-stream" } },
              );
            }
            return Response.json(
              anthropic
                ? {
                    id: "msg_test", type: "message", role: "assistant", model: modelName,
                    content: [], stop_reason: "end_turn", stop_sequence: null,
                    usage: { input_tokens: 1, output_tokens: 1 },
                  }
                : {
                    id: "chatcmpl_test", object: "chat.completion", created: 0, model: modelName,
                    choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
                    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
                  },
            );
          };
          const credentials = { apiKey: "test-api-key", baseURL: config.baseURL };
          const provider = anthropic ? new AnthropicModelProvider(credentials) : new OpenAIModelProvider(credentials);
          if (provider instanceof AnthropicModelProvider) {
            provider._client = new Anthropic({ ...credentials, fetch: fetchRequest });
          } else {
            provider._client = new OpenAI({ ...credentials, fetch: fetchRequest });
          }
          const { effort, ...overrides } = requestCase.options ?? {};
          const options = {
            ...getDefaultModelOptions(modelName, config.providerType),
            ...overrides,
            ...(effort ? (anthropic ? { output_config: { effort } } : { reasoning_effort: effort }) : {}),
          };
          const model = new Model(modelName, provider, options);
          const messages = structuredClone(HISTORY);
          if (stream) {
            for await (const message of model.stream({ messages })) {
              expect(message.role).toBe("assistant");
            }
          } else {
            await model.invoke({ messages });
          }

          expect(requestURL).toBe(`${expected.baseURL}${anthropic ? "/v1/messages" : "/chat/completions"}`);
          expect(body.model).toBe(modelName);
          expect(body.max_tokens).toBe(16 * 1024);
          expect(body.stream).toBe(stream ? true : undefined);
          expect(body.service_tier).toBeUndefined();
          expect(body.reasoning_split).toBeUndefined();
          if (modelName === "MiniMax-M3.1-Flash") {
            expect(body.thinking).toEqual({ type: "adaptive" });
          } else if (requestCase.options?.thinking) {
            expect(body.thinking).toEqual(requestCase.options.thinking);
          } else if (modelName === "MiniMax-M3") {
            expect(body.thinking).toEqual(anthropic ? undefined : { type: "adaptive" });
          } else if (modelName === "MiniMax-M2.7") {
            expect(body.thinking).toBeUndefined();
          } else {
            expect(body.thinking).toMatchObject({ type: "enabled" });
          }
          expect(body.reasoning_effort).toBe(anthropic ? undefined : effort);
          expect(body.output_config).toEqual(anthropic && effort ? { effort } : undefined);
          const wireHistory = JSON.stringify(body.messages);
          for (const value of ["Prior reasoning", "call_test", "inspect", "Image inspected", "https://example.com/image.png", "Continue"]) {
            expect(wireHistory).toContain(value);
          }
          expect(messages).toEqual(HISTORY);
        });
      }
    }
  }
});
