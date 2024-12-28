import Ollama, { ClientOptions } from "@ollama/sdk";
import { zodToJsonSchema } from "zod-to-json-schema";
import { LogLine } from "../../types/log";
import { AvailableModel, OllamaTransformedResponse } from "../../types/model";
import { LLMCache } from "../cache/LLMCache";
import { ChatCompletionOptions, LLMClient } from "./LLMClient";

export class OllamaClient extends LLMClient {
  public type = "ollama" as const;
  private client: Ollama;
  private cache: LLMCache | undefined;
  public logger: (message: LogLine) => void;
  private enableCaching: boolean;
  public clientOptions: ClientOptions;

  constructor(
    logger: (message: LogLine) => void,
    enableCaching = false,
    cache: LLMCache | undefined,
    modelName: AvailableModel,
    clientOptions?: ClientOptions,
  ) {
    super(modelName);
    this.clientOptions = clientOptions;
    this.client = new Ollama(clientOptions);
    this.logger = logger;
    this.cache = cache;
    this.enableCaching = enableCaching;
    this.modelName = modelName;
  }

  async createChatCompletion<T = OllamaTransformedResponse>(
    optionsInitial: ChatCompletionOptions,
    retries: number = 3,
  ): Promise<T> {
    let options: Partial<ChatCompletionOptions> = optionsInitial;

    const { image, requestId, ...optionsWithoutImageAndRequestId } = options;

    this.logger({
      category: "ollama",
      message: "creating chat completion",
      level: 1,
      auxiliary: {
        options: {
          value: JSON.stringify({
            ...optionsWithoutImageAndRequestId,
            requestId,
          }),
          type: "object",
        },
        modelName: {
          value: this.modelName,
          type: "string",
        },
      },
    });

    const cacheOptions = {
      model: this.modelName,
      messages: options.messages,
      temperature: options.temperature,
      top_p: options.top_p,
      frequency_penalty: options.frequency_penalty,
      presence_penalty: options.presence_penalty,
      image: image,
      response_model: options.response_model,
    };

    if (this.enableCaching) {
      const cachedResponse = await this.cache.get<T>(
        cacheOptions,
        options.requestId,
      );
      if (cachedResponse) {
        this.logger({
          category: "llm_cache",
          message: "LLM cache hit - returning cached response",
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
            cachedResponse: {
              value: JSON.stringify(cachedResponse),
              type: "object",
            },
          },
        });
        return cachedResponse;
      } else {
        this.logger({
          category: "llm_cache",
          message: "LLM cache miss - no cached response found",
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
          },
        });
      }
    }

    if (options.image) {
      const screenshotMessage = {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: {
              url: `data:image/jpeg;base64,${options.image.buffer.toString("base64")}`,
            },
          },
          ...(options.image.description
            ? [{ type: "text", text: options.image.description }]
            : []),
        ],
      };

      options.messages.push(screenshotMessage);
    }

    let responseFormat = undefined;
    if (options.response_model) {
      try {
        const parsedSchema = JSON.stringify(
          zodToJsonSchema(options.response_model.schema),
        );
        options.messages.push({
          role: "user",
          content: `Respond in this zod schema format:\n${parsedSchema}\n

        Do not include any other text, formatting or markdown in your output. Do not include \`\`\` or \`\`\`json in your response. Only the JSON object itself.`,
        });
      } catch (error) {
        this.logger({
          category: "ollama",
          message: "Failed to parse response model schema",
          level: 0,
        });

        if (retries > 0) {
          return this.createChatCompletion(
            options as ChatCompletionOptions,
            retries - 1,
          );
        }

        throw error;
      }
    }

    const { response_model, ...ollamaOptions } = {
      ...optionsWithoutImageAndRequestId,
      model: this.modelName,
    };

    this.logger({
      category: "ollama",
      message: "creating chat completion",
      level: 1,
      auxiliary: {
        ollamaOptions: {
          value: JSON.stringify(ollamaOptions),
          type: "object",
        },
      },
    });

    const formattedMessages = options.messages.map((message) => {
      if (Array.isArray(message.content)) {
        const contentParts = message.content.map((content) => {
          if ("image_url" in content) {
            return {
              image_url: {
                url: content.image_url.url,
              },
              type: "image_url",
            };
          } else {
            return {
              text: content.text,
              type: "text",
            };
          }
        });

        if (message.role === "system") {
          return {
            ...message,
            role: "system",
            content: contentParts.filter(
              (content) => content.type === "text",
            ),
          };
        } else if (message.role === "user") {
          return {
            ...message,
            role: "user",
            content: contentParts,
          };
        } else {
          return {
            ...message,
            role: "assistant",
            content: contentParts.filter(
              (content) => content.type === "text",
            ),
          };
        }
      }

      return {
        role: "user",
        content: message.content,
      };
    });

    const body = {
      ...ollamaOptions,
      model: this.modelName,
      messages: formattedMessages,
      response_format: responseFormat,
      stream: false,
      tools: options.tools?.filter((tool) => "function" in tool),
    };

    const response = await this.client.chat.completions.create(body);

    // For O1 models, we need to parse the tool call response manually and add it to the response.
    if (options.response_model) {
      try {
        const parsedContent = JSON.parse(response.choices[0].message.content);

        response.choices[0].message.tool_calls = [
          {
            function: {
              name: parsedContent["name"],
              arguments: JSON.stringify(parsedContent["arguments"]),
            },
            type: "function",
            id: "-1",
          },
        ];
        response.choices[0].message.content = null;
      } catch (error) {
        this.logger({
          category: "ollama",
          message: "Failed to parse tool call response",
          level: 0,
          auxiliary: {
            error: {
              value: error.message,
              type: "string",
            },
            content: {
              value: response.choices[0].message.content,
              type: "string",
            },
          },
        });

        if (retries > 0) {
          return this.createChatCompletion(
            options as ChatCompletionOptions,
            retries - 1,
          );
        }

        throw error;
      }
    }

    this.logger({
      category: "ollama",
      message: "response",
      level: 1,
      auxiliary: {
        response: {
          value: JSON.stringify(response),
          type: "object",
        },
        requestId: {
          value: requestId,
          type: "string",
        },
      },
    });

    if (options.response_model) {
      const extractedData = response.choices[0].message.content;
      const parsedData = JSON.parse(extractedData);

      if (!validateZodSchema(options.response_model.schema, parsedData)) {
        if (retries > 0) {
          return this.createChatCompletion(
            options as ChatCompletionOptions,
            retries - 1,
          );
        }

        throw new Error("Invalid response schema");
      }

      if (this.enableCaching) {
        this.cache.set(
          cacheOptions,
          {
            ...parsedData,
          },
          options.requestId,
        );
      }

      return parsedData;
    }

    if (this.enableCaching) {
      this.logger({
        category: "llm_cache",
        message: "caching response",
        level: 1,
        auxiliary: {
          requestId: {
            value: options.requestId,
            type: "string",
          },
          cacheOptions: {
            value: JSON.stringify(cacheOptions),
            type: "object",
          },
          response: {
            value: JSON.stringify(response),
            type: "object",
          },
        },
      });
      this.cache.set(cacheOptions, response, options.requestId);
    }

    return response as T;
  }
}
