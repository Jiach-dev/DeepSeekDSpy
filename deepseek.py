import dspy
import os

#Define an enumeration of valid liteLLM model names for DeepSeek models valid in the DSPy API
class Model(str):
    DEEPSEEK_REASONER = "deepseek/deepseek-reasoner"
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"

# get the API key from the environment
deepseek_api_key = os.getenv("DEEPSEEKAPIKEY")
if not deepseek_api_key:
    raise ValueError("Please set the DEEPSEEKAPIKEY environment variable")

# Set up DSPy
lm = dspy.LM(Model.DEEPSEEK_REASONER, api_key=deepseek_api_key, api_base='https://api.deepseek.com')
dspy.configure(lm=lm)

class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")

class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
            section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)

draft_article = DraftArticle()
article = draft_article(topic="Asynchronous Programming in Python")

#print the article markdown
print(article)

# Print out last 10 LLM calls for debugging purposes
dspy.inspect_history(n=10)
