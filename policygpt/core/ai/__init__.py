"""policygpt.core.ai — AI provider abstraction layer.

Sub-packages:
    providers/   — OpenAI and Bedrock provider implementations
    pricing/     → moved to policygpt.observability.pricing

Public API:
    base.AIService           — Protocol all providers must satisfy
    base.AIRequestTooLargeError
    registry.build_ai_service — factory: Config → AIService
"""
