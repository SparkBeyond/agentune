import logging

import httpx
import pytest

from agentune.analyze.core import types
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.dedup.llm_based_deduplicator import (
    LLMBasedDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query

_logger = logging.getLogger(__name__)

@pytest.fixture
async def llm_with_spec_for_dedup(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a real LLM for end-to-end testing."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'o3')
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec

@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_based_deduplicator(llm_with_spec_for_dedup: LLMWithSpec) -> None:
    """Test query semantic dedup."""
    query_list = [Query('q1', 'Did the seller offer a discount?', types.boolean),
                  Query('q2', 'Which next step did the seller and buyer decide on?', types.string),
                  Query('q3', '''Did the seller ask about the buyer's needs or pain points?''', types.boolean),
                  Query('q4', 'Did the seller explain the value/benefits (not just features)?', types.boolean),
                  Query('q5', 'Did the seller create urgency (limited-time offer, limited stock)?', types.boolean),
                  Query('q6', 'Did the seller handle objections or concerns (e.g., price, features)?', types.boolean),
                  Query('q7', "Rate the seller's enthusiasm on a 1 to 5 scale.", types.float64),
                  Query('q8', 'Did the seller set a follow-up action (meeting, trial, contract)?', types.boolean),
                  Query('q9', 'Did the seller gain commitment from the buyer (yes/no, next step)?', types.boolean),
                  Query('q10', 'What is the count of product name mentions by the seller?', types.int32),
                  Query('q11', 'Was any special pricing or promotional deal mentioned by the seller?', types.boolean),
                  Query('q12', 'Did the customer make any kind of decision or agreement during the call?', types.boolean),
                  Query('q13', 'Did the salesperson try to pressure the buyer by stressing time or availability limits?', types.boolean),
                  Query('q14', 'Did the salesperson address any doubts or pushback from the customer?', types.boolean),
                  Query('q15', 'Did the salesperson highlight how the product would help the buyer, beyond listing features?', types.boolean),
                  Query('q16', '''Did the salesperson try to understand the customer's challenges or requirements?''', types.boolean),
                  Query('q17', 'Was there an agreed next step scheduled by the salesperson?', types.boolean),
                  Query('q18', 'How many times did the seller mention the product name?', types.int32),
                  Query('q19', 'On a scale of 1 to 5, how enthusiastic did the seller sound?', types.float64),
                  Query('q20', 'What follow-up action was agreed upon?', types.string),
                  Query('q21', 'How many objections did the customer raise?', types.int32),
                  Query('q22', 'What percentage of the call time did the customer speak?', types.float64),
                  Query('q23', 'Which competitor did the customer mention, if any?', types.string),
                  Query('q24', 'Did the salesperson suggest a reduced price or promotion?', types.boolean)]

    # define pairs (at least one must be in kept_ids)
    pairs = [
        ('q3', 'q16'),
        ('q4', 'q15'),
        ('q5', 'q13'),
        ('q6', 'q14'),
        ('q8', 'q17'),
        ('q9', 'q12'),
        ('q18', 'q10'),
        ('q19', 'q7'),
        ('q20', 'q2'),
    ]

    # Define groups (only one should be kept from each group)
    groups = [['q1', 'q11', 'q24']]

    # Define singletons (all must be in kept_ids)
    singletons = ['q21', 'q22', 'q23']

    deduplicator = LLMBasedDeduplicator(llm_with_spec_for_dedup)
    queries_to_keep = await deduplicator.deduplicate(query_list)

    # Extract kept IDs
    kept_ids = {q.name for q in queries_to_keep}
    _logger.debug(f'Kept query IDs: {kept_ids}')

    # Check pairs - exactly one from each pair should be kept (allow 2 false positives)
    false_positives = 0
    for a, b in pairs:
        a_kept = a in kept_ids
        b_kept = b in kept_ids

        # Ensure at least one is kept
        assert a_kept or b_kept, f'Neither {a} nor {b} was kept!'

        # Count false positives (both kept)
        if a_kept and b_kept:
            false_positives += 1
            _logger.debug(f'False positive: both {a} and {b} were kept')

    # Check groups
    for group in groups:
        kept_in_group = [qid for qid in group if qid in kept_ids]
        if len(kept_in_group) > 1:
            false_positives += (len(kept_in_group) - 1)
            _logger.debug(f'False positive: multiple kept in group {group}: {kept_in_group}')
        assert len(kept_in_group) >= 1, f'No queries kept in group {group}'

    # Allow up to 2 false positives
    assert false_positives <= 2, f'Too many false positives: {false_positives} pairs had both members kept (max allowed: 2)'

    # Check singletons
    for s in singletons:
        assert s in kept_ids, f'Singleton {s} was not kept!'

    _logger.debug('Dedup result passed validation!')

