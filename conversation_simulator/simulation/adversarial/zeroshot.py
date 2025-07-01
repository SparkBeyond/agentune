"""Zero-shot adversarial tester implementation using a language model."""

import logging
import random
from typing import override

from attrs import field, frozen
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                                  SystemMessagePromptTemplate)
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda

from .base import AdversarialTest, AdversarialTester
from .prompts import (
    HUMAN_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    create_comparison_prompt_inputs,
)

logger = logging.getLogger(__name__)

@frozen
class ZeroShotAdversarialTester(AdversarialTester):
    """Zero-shot adversarial tester using a language model and a structured parser.
    
    Attributes:
        model: The language model to use for evaluation.
        max_concurrency: The maximum number of concurrent requests to the model.
        random_seed: Optional seed for reproducible random number generation.
    """

    model: BaseChatModel
    max_concurrency: int = 50
    random_seed: int = 0

    _random: random.Random = field(init=False)
    @_random.default
    def _random_default(self) -> random.Random:
        return random.Random(self.random_seed)
    
    _chain: Runnable = field(init=False)
    @_chain.default
    def _create_adversarial_chain(self) -> Runnable:
        """Creates the LangChain runnable for adversarial evaluation."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
        ])

        label_extractor = RunnableLambda(self._extract_label)
        return prompt | self.model | JsonOutputParser() | label_extractor

    @staticmethod
    def _extract_label(output: dict) -> str | None:
        """Extracts the identified conversation label from the model output."""
        identified_label = output.get("real_conversation")
        if not isinstance(identified_label, str) or identified_label not in ("A", "B"):
            logger.warning(f"LLM returned invalid value: {identified_label}")
            return None
        return identified_label

    @override
    async def identify_real_conversations(
        self,
        instances: tuple[AdversarialTest, ...],
        return_exceptions: bool = True
    ) -> tuple[bool | None | Exception, ...]:
        if not instances:
            return tuple()

        # Prepare inputs and track which indices are valid
        prompt_inputs = []
        valid_indices = []
        real_labels = []
        results: list[bool | None | Exception] = [None] * len(instances)

        for i, instance in enumerate(instances):
            if not instance.real_conversation.messages or not instance.simulated_conversation.messages:
                continue  # Skip empty conversations, leaving as None

            is_real_a = self._random.choice([True, False])
            prompt_inputs.append(
                create_comparison_prompt_inputs(instance.real_conversation, instance.simulated_conversation, is_real_a)
            )
            real_labels.append("A" if is_real_a else "B")
            valid_indices.append(i)

        if not prompt_inputs:
            return tuple(results)

        # Process valid conversations in batch
        identified_labels = await self._chain.abatch(prompt_inputs, max_concurrency=self.max_concurrency, return_exceptions=return_exceptions)
        
        for idx, identified_label, real_label in zip(valid_indices, identified_labels, real_labels):
            if identified_label is None:
                continue
            results[idx] = identified_label == real_label

        return tuple(results)
