MAX_SUMMARY_LENGTH = 100
PROMPT_TEMPLATE = f"""
You are tasked with creating a concise summary of a conversation history between an AI entity and a human user.
The provided conversation extract represents a fixed context window and may be incomplete.
AI responses are identified by the 'assistant' role designation.
The AI entity can execute functions, with results visible in 'function' role messages.
AI message content represents internal processing and remains invisible to users.
User-visible AI communications are exclusively transmitted via 'send_message'.
Human inputs and system events are designated with the 'user' role.
The 'user' role encompasses critical system events including authentication and automated heartbeat processes.
Generate a first-person narrative summarizing the conversation from the AI's perspective.
Maintain summary length below {MAX_SUMMARY_LENGTH} words - strict adherence required.
Provide summary content exclusively without additional commentary.
"""
