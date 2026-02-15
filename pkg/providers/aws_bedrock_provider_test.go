package providers

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestMessagesToConverseInput(t *testing.T) {
	p := NewAWSBedrockProvider(".")

	tests := []struct {
		name                 string
		messages             []Message
		expectedSystemCount  int
		expectedMessageCount int
		expectedRoles        []types.ConversationRole
	}{
		{
			name: "single system and user message",
			messages: []Message{
				{Role: "system", Content: "System prompt"},
				{Role: "user", Content: "Hello"},
			},
			expectedSystemCount:  1,
			expectedMessageCount: 1,
			expectedRoles:        []types.ConversationRole{types.ConversationRoleUser},
		},
		{
			name: "alternating roles",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
				{Role: "user", Content: "How are you?"},
			},
			expectedSystemCount:  0,
			expectedMessageCount: 3,
			expectedRoles:        []types.ConversationRole{types.ConversationRoleUser, types.ConversationRoleAssistant, types.ConversationRoleUser},
		},
		{
			name: "consecutive user messages should merge",
			messages: []Message{
				{Role: "user", Content: "Message 1"},
				{Role: "user", Content: "Message 2"},
			},
			expectedSystemCount:  0,
			expectedMessageCount: 1,
			expectedRoles:        []types.ConversationRole{types.ConversationRoleUser},
		},
		{
			name: "user message then tool result should merge",
			messages: []Message{
				{Role: "user", Content: "Run tool"},
				{Role: "tool", Content: "Result", ToolCallID: "call_1"},
			},
			expectedSystemCount:  0,
			expectedMessageCount: 1,
			expectedRoles:        []types.ConversationRole{types.ConversationRoleUser},
		},
		{
			name: "complex sequence with merging",
			messages: []Message{
				{Role: "system", Content: "Sys 1"},
				{Role: "system", Content: "Sys 2"},
				{Role: "user", Content: "User 1"},
				{Role: "assistant", Content: "Assist 1"},
				{Role: "assistant", Content: "Assist 2"},
				{Role: "user", Content: "User 2"},
				{Role: "tool", Content: "Res 1", ToolCallID: "call_a"},
				{Role: "tool", Content: "Res 2", ToolCallID: "call_b"},
			},
			expectedSystemCount:  2,
			expectedMessageCount: 3, // User 1, Assistant 1+2, User 2+Res 1+Res 2
			expectedRoles:        []types.ConversationRole{types.ConversationRoleUser, types.ConversationRoleAssistant, types.ConversationRoleUser},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := p.messagesToConverseInput(tt.messages, "some-model", nil)

			if len(input.System) != tt.expectedSystemCount {
				t.Errorf("expected %d system blocks, got %d", tt.expectedSystemCount, len(input.System))
			}

			if len(input.Messages) != tt.expectedMessageCount {
				t.Errorf("expected %d messages, got %d", tt.expectedMessageCount, len(input.Messages))
			}

			for i, role := range tt.expectedRoles {
				if i < len(input.Messages) && input.Messages[i].Role != role {
					t.Errorf("expected role %s at index %d, got %s", role, i, input.Messages[i].Role)
				}
			}
		})
	}
}
