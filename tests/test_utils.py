"""Basic unit tests for utility functions."""

from src.utils import extract_final_output, extract_thinking


def test_extract_final_output():
    text = "Hello <think> secret </think> world"
    assert extract_final_output(text) == "world"

    text_no_think = "No tags here"
    assert extract_final_output(text_no_think) == "No tags here"


def test_extract_thinking():
    text = "A <think>foo</think> B <think>bar</think>"
    assert extract_thinking(text) == "foo\nbar"
