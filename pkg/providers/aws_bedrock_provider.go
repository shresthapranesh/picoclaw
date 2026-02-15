package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type AWSBedrockProvider struct {
	command   string
	workspace string
}

func NewAWSBedrockProvider(workspace string) *AWSBedrockProvider {
	return &AWSBedrockProvider{
		command:   "awsbedrock",
		workspace: workspace,
	}
}

func (p *AWSBedrockProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]interface{}) (*LLMResponse, error) {
	BedrockRuntimeClient, err := getBedrockRuntimeClient()
	if err != nil {
		return nil, err
	}
	converseInput := p.messagesToConverseInput(messages, model, tools)
	response, err := BedrockRuntimeClient.Converse(ctx, &converseInput)
	if err != nil {
		processError(err, model)
		return nil, err
	}
	return p.parseAWSBedrockResponse(response)
}

func (p *AWSBedrockProvider) GetDefaultModel() string {
	return "anthropic.claude-haiku-4-5-20251001-v1:0"
}

func (p *AWSBedrockProvider) messagesToConverseInput(messages []Message, model string, tools []ToolDefinition) bedrockruntime.ConverseInput {
	var systemBlocks []types.SystemContentBlock
	var conversationMessages []types.Message

	for _, msg := range messages {
		if msg.Role == "system" {
			systemBlocks = append(systemBlocks, &types.SystemContentBlockMemberText{Value: msg.Content})
			continue
		}

		var bedrockRole types.ConversationRole
		var contentBlocks []types.ContentBlock
		switch msg.Role {
		case "user":
			bedrockRole = types.ConversationRoleUser
			if msg.ToolCallID != "" {
				contentBlocks = append(contentBlocks, &types.ContentBlockMemberToolResult{
					Value: types.ToolResultBlock{
						ToolUseId: aws.String(msg.ToolCallID),
						Content: []types.ToolResultContentBlock{
							&types.ToolResultContentBlockMemberText{Value: msg.Content},
						},
					},
				})
			} else {
				contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: msg.Content})
			}
		case "assistant":
			bedrockRole = types.ConversationRoleAssistant
			if len(msg.ToolCalls) > 0 {
				if msg.Content != "" {
					contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: msg.Content})
				}
				for _, tc := range msg.ToolCalls {
					name := tc.Name
					if name == "" && tc.Function != nil {
						name = tc.Function.Name
					}
					args := tc.Arguments
					if len(args) == 0 && tc.Function != nil && tc.Function.Arguments != "" {
						var parsed map[string]interface{}
						if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err == nil {
							args = parsed
						}
					}
					contentBlocks = append(contentBlocks, &types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String(tc.ID),
							Name:      aws.String(name),
							Input:     document.NewLazyDocument(args),
						},
					})
				}
			} else {
				contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: msg.Content})
			}
		case "tool":
			bedrockRole = types.ConversationRoleUser
			contentBlocks = append(contentBlocks, &types.ContentBlockMemberToolResult{
				Value: types.ToolResultBlock{
					ToolUseId: aws.String(msg.ToolCallID),
					Content: []types.ToolResultContentBlock{
						&types.ToolResultContentBlockMemberText{Value: msg.Content},
					},
				},
			})
		default:
			continue
		}

		if len(conversationMessages) > 0 && conversationMessages[len(conversationMessages)-1].Role == bedrockRole {
			lastMsg := &conversationMessages[len(conversationMessages)-1]
			lastMsg.Content = append(lastMsg.Content, contentBlocks...)
		} else {
			conversationMessages = append(conversationMessages, types.Message{
				Role:    bedrockRole,
				Content: contentBlocks,
			})
		}
	}

	input := bedrockruntime.ConverseInput{
		ModelId:  aws.String(model),
		Messages: conversationMessages,
	}

	if len(systemBlocks) > 0 {
		input.System = systemBlocks
	}

	if len(tools) > 0 {
		var toolConfigs []types.Tool
		for _, tool := range tools {
			toolConfigs = append(toolConfigs, &types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String(tool.Function.Name),
					Description: aws.String(tool.Function.Description),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(tool.Function.Parameters),
					},
				},
			})
		}
		input.ToolConfig = &types.ToolConfiguration{
			Tools: toolConfigs,
		}
	}

	return input
}

// parseAWSBedrockResponse parses the JSON output from the AWS Bedrock API.
func (p *AWSBedrockProvider) parseAWSBedrockResponse(response *bedrockruntime.ConverseOutput) (*LLMResponse, error) {
	outputMsg, ok := response.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return nil, fmt.Errorf("unexpected output type")
	}
	message := outputMsg.Value

	var content strings.Builder
	var toolCalls []ToolCall

	for _, block := range message.Content {
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			if content.Len() > 0 {
				content.WriteString("\n")
			}
			content.WriteString(b.Value)
		case *types.ContentBlockMemberToolUse:
			toolUse := b.Value
			args := map[string]interface{}{}
			if toolUse.Input != nil {
				if inputBytes, err := toolUse.Input.MarshalSmithyDocument(); err == nil {
					json.Unmarshal(inputBytes, &args)
				}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        *toolUse.ToolUseId,
				Name:      *toolUse.Name,
				Arguments: args,
			})
		}
	}

	finishReason := "stop"
	if response.StopReason != "" {
		switch response.StopReason {
		case types.StopReasonToolUse:
			finishReason = "tool_calls"
		case types.StopReasonMaxTokens:
			finishReason = "length"
		case types.StopReasonEndTurn:
			finishReason = "stop"
		}
	}

	var usage *UsageInfo
	if response.Usage != nil {
		usage = &UsageInfo{
			PromptTokens:     int(*response.Usage.InputTokens),
			CompletionTokens: int(*response.Usage.OutputTokens),
			TotalTokens:      int(*response.Usage.TotalTokens),
		}
	}

	return &LLMResponse{
		Content:      content.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

func getBedrockRuntimeClient() (*bedrockruntime.Client, error) {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	client := bedrockruntime.NewFromConfig(cfg)
	return client, nil
}

func processError(err error, modelId string) {
	errMsg := err.Error()
	if strings.Contains(errMsg, "no such host") {
		fmt.Printf(`The Bedrock service is not available in the selected region.
                    Please double-check the service availability for your region at
                    https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/.\n`)
	} else if strings.Contains(errMsg, "Could not resolve the foundation model") {
		fmt.Printf(`Could not resolve the foundation model from model identifier: \"%v\".
                    Please verify that the requested model exists and is accessible
                    within the specified region.\n
                    `, modelId)
	} else {
		fmt.Printf("Couldn't invoke model: \"%v\". Here's why: %v\n", modelId, err)
	}
}
