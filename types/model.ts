import type { ClientOptions as AnthropicClientOptions } from "@anthropic-ai/sdk";
import type { ClientOptions as OpenAIClientOptions } from "openai";
import type { ClientOptions as OllamaClientOptions } from "@ollama/sdk"; // Import OllamaClientOptions
import { ChatCompletionTool as OpenAITool } from "openai/resources";
import { z } from "zod";

export const AvailableModelSchema = z.enum([
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4o-2024-08-06",
  "claude-3-5-sonnet-latest",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-sonnet-20240620",
  "o1-mini",
  "o1-preview",
  "gemma2:2b",
]);

export type AvailableModel = z.infer<typeof AvailableModelSchema>;

export type ModelProvider = "openai" | "anthropic" | "ollama"; // Add ollama to ModelProvider

export type ClientOptions = OpenAIClientOptions | AnthropicClientOptions | OllamaClientOptions; // Include OllamaClientOptions

export type ToolCall = OpenAITool;

export type AnthropicTransformedResponse = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls: {
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }[];
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type OllamaTransformedResponse = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls: {
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }[];
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export interface AnthropicJsonSchemaObject {
  definitions?: {
    MySchema?: { properties?: Record<string, unknown>; required?: string[] };
  };
  properties?: Record<string, unknown>;
  required?: string[];
}
