
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import language_tool_python

# Load pretrained T5 model
MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Initialize grammar and spelling checker
grammar_tool = language_tool_python.LanguageTool("en-US")


def preprocess_text(text):
    """Clean and normalize input text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def paraphrase_text(text):
    """Generate paraphrased text using T5."""

    text = preprocess_text(text)

    if not text:
        raise ValueError("Input text cannot be empty.")

    input_text = f"paraphrase: {text} </s>"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.strip()


def grammar_spelling_check(text):
    """Identify possible grammar and spelling issues."""

    matches = grammar_tool.check(text)

    if not matches:
        print("No grammar or spelling issues detected.")
        return

    print("\nGrammar and Spelling Suggestions:")

    for match in matches:
        print("Issue:", match.message)
        print("Suggestion:", match.replacements[:3])


def fluency_check(text):
    """Perform basic fluency checks."""

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        print("Fluency check: No complete sentences found.")
        return

    average_length = sum(
        len(sentence.split()) for sentence in sentences
    ) / len(sentences)

    print("\nFluency Assessment:")

    if average_length > 30:
        print("Suggestion: Consider shortening long sentences.")
    elif average_length < 3:
        print("Suggestion: Check whether sentences are too short.")
    else:
        print("Sentence length appears reasonable.")

    print("Sentence count:", len(sentences))


def main():

    print("AI POWERED PARAPHRASING TOOL")
    print("=" * 35)

    input_text = input("\nEnter your text: ")

    try:
        output = paraphrase_text(input_text)

        print("\nOriginal Text:")
        print(input_text)

        print("\nParaphrased Text:")
        print(output)

        grammar_spelling_check(output)
        fluency_check(output)

    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
