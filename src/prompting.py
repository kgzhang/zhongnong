"""Q/A-format few-shot prompting with cross-chunk context."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data import ExampleData
from src.format_handler import FormatHandler


@dataclass
class PromptTemplateStructured:
    """A structured prompt template with description and few-shot examples."""

    description: str
    examples: list[ExampleData] = field(default_factory=list)


@dataclass
class QAPromptGenerator:
    """Generates Q/A-format prompts from a template and format handler."""

    template: PromptTemplateStructured
    format_handler: FormatHandler
    examples_heading: str = "Examples"
    question_prefix: str = "Q: "
    answer_prefix: str = "A: "

    def format_example_as_text(self, example: ExampleData) -> str:
        """Format a single example as Q/A text."""
        q = f"{self.question_prefix}{example.text}"
        a = (
            f"{self.answer_prefix}"
            f"{self.format_handler.format_extraction_example(example.extractions)}"
        )
        return f"{q}\n{a}"

    def render(
        self, question: str, additional_context: str | None = None
    ) -> str:
        """Render the full prompt.

        Structure: description, optional context, optional examples,
        then the question with an empty answer suffix for the model to fill.
        """
        parts: list[str] = [self.template.description]

        # Optional context section
        if additional_context:
            parts.append(additional_context)

        # Optional few-shot examples section
        if self.template.examples:
            lines: list[str] = [self.examples_heading]
            for ex in self.template.examples:
                lines.append(self.format_example_as_text(ex))
            parts.append("\n".join(lines))

        # Question with empty answer prefix
        parts.append(f"{self.question_prefix}{question}\n{self.answer_prefix}")

        return "\n\n".join(parts)


class PromptBuilder:
    """Builds prompts by delegating to a QAPromptGenerator.

    Stateless per document — each invocation produces a self-contained prompt.
    """

    def __init__(self, generator: QAPromptGenerator) -> None:
        self.generator = generator

    def build_prompt(
        self,
        chunk_text: str,
        document_id: str,
        additional_context: str | None = None,
    ) -> str:
        """Build a prompt for the given chunk text.

        *document_id* is accepted for API compatibility with
        ``ContextAwarePromptBuilder`` but not used here.
        """
        return self.generator.render(
            question=chunk_text,
            additional_context=additional_context,
        )


class ContextAwarePromptBuilder(PromptBuilder):
    """Builds prompts with cross-chunk context awareness.

    Tracks the previous chunk per document and, when *context_window_chars*
    is set, prepends the tail of the prior chunk as ``[Previous text]: …``
    context to help the model maintain coherence across chunk boundaries.
    """

    CONTEXT_PREFIX = "[Previous text]: ..."

    def __init__(
        self,
        generator: QAPromptGenerator,
        context_window_chars: int | None = None,
    ) -> None:
        super().__init__(generator)
        self.context_window_chars = context_window_chars
        self._prev_chunks: dict[str, str] = {}

    def build_prompt(
        self,
        chunk_text: str,
        document_id: str,
        additional_context: str | None = None,
    ) -> str:
        """Build prompt with cross-chunk context."""
        effective_context = self._build_effective_context(
            document_id, additional_context
        )
        self._update_state(document_id, chunk_text)
        return self.generator.render(
            question=chunk_text,
            additional_context=effective_context,
        )

    def _build_effective_context(
        self,
        doc_id: str,
        additional_context: str | None,
    ) -> str | None:
        """Combine previous-chunk context with *additional_context*.

        Returns ``None`` when there is nothing to add.
        """
        prev_chunk = self._prev_chunks.get(doc_id)
        prev_context: str | None = None

        if prev_chunk is not None and self.context_window_chars is not None:
            tail = prev_chunk[-self.context_window_chars :]
            prev_context = f"{self.CONTEXT_PREFIX}{tail}"

        if prev_context and additional_context:
            return f"{prev_context}\n{additional_context}"
        if prev_context:
            return prev_context
        if additional_context:
            return additional_context
        return None

    def _update_state(self, doc_id: str, chunk_text: str) -> None:
        """Store the current chunk for the next call.

        Only tracks state when *context_window_chars* is set.
        """
        if self.context_window_chars is not None:
            self._prev_chunks[doc_id] = chunk_text
