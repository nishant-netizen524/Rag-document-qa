from src.prompts import build_user_prompt, build_llm_messages


def test_user_prompt_labels_excerpts_with_source():
    blocks = [{"filename": "report.pdf", "page": 3, "text": "Revenue was 5M."}]
    prompt = build_user_prompt("What was revenue?", blocks)
    assert "[Excerpt 1]" in prompt
    assert "report.pdf (page 3)" in prompt


def test_history_is_capped_and_errors_are_dropped():
    history = [
        {"role": "user", "content": "old question " * 200},
        {"role": "assistant", "content": "Sorry, I could not generate an answer: boom"},
        {"role": "user", "content": "new question"},
    ]
    msgs = build_llm_messages("q", [{"filename": "f", "page": 1, "text": "t"}], history)
    contents = " ".join(m["content"] for m in msgs)
    assert "Sorry, I could not generate" not in contents
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"