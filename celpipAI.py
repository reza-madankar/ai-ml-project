from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
from pydantic import BaseModel, Field


class Question(BaseModel):
    prompt: str
    items: list[str] = Field(min_length=3, max_length=3)


class SampleAnswer(BaseModel):
    title: str
    content: str

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class CelpipWriting(BaseModel):
    question: Question
    sampleAnswer: list[SampleAnswer] = Field(min_length=3, max_length=3)


prompt = """Generate a CELPIP Writing Task 1 (Writing an Email). Return the output in strict JSON format with the following structure:
{
  "question": {
    "prompt": "[Email scenario with recipient and context]",
    "items": ["[Bullet point 1]", "[Bullet point 2]", "[Bullet point 3]"]
  },
  "sampleAnswer": [
    {"title": "CLB 4-5", "content": "[Response text, 140–160 words]"},
    {"title": "CLB 6-7", "content": "[Response text, 160–180 words]"},
    {"title": "CLB 8-10", "content": "[Response text, 180–200 words]"}
  ]
}
Requirements:
1. **Question**:
   - Explain a situation and clearly it in the title in 30 - 50 words.
   - Do not use how many minutes you need and how many words you should write
   - Create a realistic email prompt asking the test-taker to describe a personal experience relevant to daily life in Canada.
   - Provide three distinct bullet points in the 'items' array outlining what to address.
   - Ensure the scenario is engaging, practical, and suitable.
   - items should contains 5 - 10 words and dont use prons and cons  .
2. **Sample Answers**:
**GLOBAL WRITING BAND RULES (STRICT)**:
- Return exactly three answers: CLB 4-5, CLB 6-7, CLB 8-10 with the word-count ranges below.
- Each band must differ in structure, vocabulary range, and depth — not just length.
- CLB 4-5: 2–4 grammar/punctuation errors; simple sentences; basic greeting/closing.
- CLB 6-7: at most 1–2 minor errors; clear paragraphs; appropriate formal email tone.
- CLB 8-10: no errors; sophisticated but natural tone; cohesive flow; all bullet points fully developed.
- Do NOT reuse the same subject line or opening sentence across bands.

   - **CLB 4–5** (140–160 words): short paragraphs; addresses all 3 bullets briefly; simple vocabulary.
   - **CLB 6–7** (160–180 words): clear email structure (greeting, body, closing); relevant details per bullet.
   - **CLB 8–10** (180–200 words): polished email; precise word choice; smooth transitions; complete, natural closing.
   - Each answer must address all three bullet points with proper email conventions.
3. **Constraints**:
   - Avoid overly complex or sensitive scenarios.
   - Ensure the scenario and answers are culturally appropriate for a Canadian context.
- Do NOT mention preparation time, response time, or any timing instructions in the description and content.
** The TrapStrategy from the selected difficulty level must be strictly followed when generating distractor options.
"""

message = {
    "role": "user",
    "content": prompt,
}






class CelpipWritingAI:
    def __init__(self, config_path: str = 'config.yaml'):
        load_dotenv()
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        provider = config['providers'][config['provider']]
        self.model = provider['model']
        self.client = OpenAI(
            base_url=provider['base_url'],
            api_key=os.getenv(provider['api_key_env']),
        )

    def generate(self) -> CelpipWriting:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[message],
            response_format={"type": "json_object"},
        )
        return CelpipWriting.model_validate_json(self._strip_code_fence(completion.choices[0].message.content))

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return text


if __name__ == "__main__":
    task = CelpipWritingAI().generate()

    print("QUESTION:\n" + task.question.prompt)
    for item in task.question.items:
        print(f"  - {item}")

    for answer in task.sampleAnswer:
        print(f"\n===== {answer.title} ({answer.word_count} words) =====")
        print(answer.content)