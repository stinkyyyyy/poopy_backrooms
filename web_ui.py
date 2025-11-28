# web_ui.py
import gradio as gr
import time
from config import AI_MODELS, SYSTEM_PROMPT_PAIRS, TURN_DELAY
from ai_logic import ai_turn

# Extract model names and preset names for dropdowns
MODEL_NAMES = list(AI_MODELS.keys())
PRESET_NAMES = list(SYSTEM_PROMPT_PAIRS.keys())

def format_history_for_gradio(history):
    """Convert internal conversation history to Gradio chatbot format."""
    gradio_history = []

    # Iterate through history to find user-assistant pairs or handle system messages
    # This is a simplified linear representation
    # Gradio chatbot expects [(user_msg, bot_msg), ...] or just a list of dicts for type="messages"
    # We will use type="messages" which expects [{"role": "user", "content": "msg"}, ...]

    return history

def run_conversation(ai1_model_name, ai2_model_name, prompt_preset, num_turns, user_input):
    """
    Generator function to run the conversation.
    Yields updated conversation history after each turn.
    """

    # Initialize conversation history
    # Gradio Chatbot component with type='messages' expects a list of dictionaries
    conversation_display = []

    # Internal conversation history for the AI logic (same format as main.py)
    # List of dicts: {"role": "user"|"assistant"|"system", "content": "...", "ai_name": "...", "model": "..."}
    internal_conversation = []

    # 1. Handle initial user input
    if user_input and user_input.strip():
        user_msg = {"role": "user", "content": user_input}
        internal_conversation.append(user_msg)
        conversation_display.append({"role": "user", "content": user_input})
        yield conversation_display, "Starting conversation..."

    # 2. Loop for the specified number of turns
    for turn in range(1, int(num_turns) + 1):
        status_msg = f"Turn {turn}/{num_turns}..."

        # --- AI-1 Turn ---
        ai1_name = "AI-1"
        yield conversation_display, f"{status_msg} {ai1_name} thinking..."

        # Get system prompt for AI-1
        ai1_prompt = SYSTEM_PROMPT_PAIRS[prompt_preset][ai1_name]

        # Call AI Logic
        try:
            # Note: ai_turn returns a dict or string. We need to handle it.
            # In ai_logic.py (refactored), it returns dict with 'content', 'role', 'model', 'ai_name'
            # OR result from generate_video_with_sora which is also a dict

            result1 = ai_turn(
                ai_name=ai1_name,
                conversation=internal_conversation,
                model=ai1_model_name,
                system_prompt=ai1_prompt
            )

            # Process Result
            content1 = ""
            if isinstance(result1, dict):
                content1 = result1.get("content", "")
            else:
                content1 = str(result1)

            # Update Internal History
            msg1 = {
                "role": "assistant",
                "content": content1,
                "ai_name": ai1_name,
                "model": ai1_model_name
            }
            internal_conversation.append(msg1)

            # Update Display History (Add Speaker Name)
            display_content1 = f"**{ai1_name} ({ai1_model_name}):**\n\n{content1}"
            conversation_display.append({"role": "assistant", "content": display_content1})

            yield conversation_display, f"{status_msg} {ai1_name} done."

        except Exception as e:
            error_msg = f"Error from {ai1_name}: {str(e)}"
            conversation_display.append({"role": "assistant", "content": f"**System Error:** {error_msg}"})
            yield conversation_display, "Error occurred."
            return # Stop on error

        # Small delay between AIs
        time.sleep(TURN_DELAY)

        # --- AI-2 Turn ---
        ai2_name = "AI-2"
        yield conversation_display, f"{status_msg} {ai2_name} thinking..."

        # Get system prompt for AI-2
        ai2_prompt = SYSTEM_PROMPT_PAIRS[prompt_preset][ai2_name]

        # Call AI Logic
        try:
            result2 = ai_turn(
                ai_name=ai2_name,
                conversation=internal_conversation,
                model=ai2_model_name,
                system_prompt=ai2_prompt
            )

            # Process Result
            content2 = ""
            if isinstance(result2, dict):
                content2 = result2.get("content", "")
            else:
                content2 = str(result2)

            # Update Internal History
            msg2 = {
                "role": "assistant",
                "content": content2,
                "ai_name": ai2_name,
                "model": ai2_model_name
            }
            internal_conversation.append(msg2)

            # Update Display History
            display_content2 = f"**{ai2_name} ({ai2_model_name}):**\n\n{content2}"
            conversation_display.append({"role": "assistant", "content": display_content2})

            yield conversation_display, f"{status_msg} {ai2_name} done."

        except Exception as e:
            error_msg = f"Error from {ai2_name}: {str(e)}"
            conversation_display.append({"role": "assistant", "content": f"**System Error:** {error_msg}"})
            yield conversation_display, "Error occurred."
            return # Stop on error

        # Delay before next turn loop
        time.sleep(TURN_DELAY)

    yield conversation_display, "Conversation finished."


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Liminal Backrooms Web") as demo:
    gr.Markdown("# Liminal Backrooms Web Interface")

    with gr.Row():
        with gr.Column(scale=1):
            ai1_dropdown = gr.Dropdown(
                choices=MODEL_NAMES,
                value=MODEL_NAMES[0] if MODEL_NAMES else None,
                label="AI-1 Model"
            )
        with gr.Column(scale=1):
            ai2_dropdown = gr.Dropdown(
                choices=MODEL_NAMES,
                value=MODEL_NAMES[1] if len(MODEL_NAMES) > 1 else MODEL_NAMES[0],
                label="AI-2 Model"
            )

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            choices=PRESET_NAMES,
            value="Backrooms",
            label="System Prompt Preset"
        )
        turns_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=1,
            step=1,
            label="Number of Turns (Loop)"
        )

    initial_prompt = gr.Textbox(
        lines=2,
        placeholder="Enter initial prompt or topic (optional)...",
        label="Initial User Input"
    )

    start_btn = gr.Button("Start Conversation", variant="primary")
    status_markdown = gr.Markdown("Ready to start.")

    chatbot = gr.Chatbot(
        type="messages",
        height=600,
        label="Conversation History"
    )

    # Event handling
    start_btn.click(
        fn=run_conversation,
        inputs=[ai1_dropdown, ai2_dropdown, preset_dropdown, turns_slider, initial_prompt],
        outputs=[chatbot, status_markdown]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
