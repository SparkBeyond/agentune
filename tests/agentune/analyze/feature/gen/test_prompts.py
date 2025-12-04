"""Tests for prompt templates and utilities."""

import pytest

from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    ACTIONABLE_QUESTIONNAIRE_PROMPT,
    CREATIVE_FEATURES_PROMPT,
    QUESTIONNAIRE_PROMPT,
    create_enrich_conversation_prompt,
    get_output_instructions,
    questionnaire_prompt_context,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import PARSER_OUT_FIELD, Query
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    ProblemDescription,
    RegressionDirection,
    RegressionProblem,
)
from agentune.core import types
from agentune.core.schema import Field


class TestQuestionnairePrompts:
    """Test questionnaire prompt templates."""

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_classification_problem_without_desired_outcome(self, prompt_template: str) -> None:
        """Test that classification problem without target_desired_outcome raises error."""
        # Create a minimal ProblemDescription with only required parameters
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem'
        )
        
        # Create a classification problem with string target (valid for classification)
        # Classes must be sorted in ascending order
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        # Call create_questionnaire_prompt with correct parameters
        examples = 'Example 1: some data\nExample 2: more data'
        instance_type = 'conversation'
        instance_description = 'customer service conversations'
        n_queries = '5'
        
        # This should raise ValueError because desired_target_outcome is None
        with pytest.raises(ValueError, match='Problem description must include target_desired_outcome'):
            context = questionnaire_prompt_context(
                examples=examples,
                problem=problem,
                instance_type=instance_type,
                instance_description=instance_description,
                n_queries=n_queries,
                existing_queries=[]
            )
            prompt_template.format(**context)

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_classification_problem_with_string_target(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with classification problem (string target)."""
        # Create ProblemDescription with target_desired_outcome (required for prompt)
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem',
            target_desired_outcome='success'
        )
        
        # Classes must be sorted in ascending order
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        # Call create_questionnaire_prompt with correct parameters
        examples = 'Example 1: some data\nExample 2: more data'
        instance_type = 'conversation'
        instance_description = 'customer service conversations'
        n_queries = '5'
        
        context = questionnaire_prompt_context(
            examples=examples,
            problem=problem,
            instance_type=instance_type,
            instance_description=instance_description,
            n_queries=n_queries,
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # Verify the prompt contains expected elements
        assert 'target = success' in result
        assert 'target != success' in result
        assert 'up to 5 questions' in result
        assert examples in result
        assert instance_type in result
        assert instance_description in result

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_classification_problem_with_integer_target(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with classification problem (integer target)."""
        problem_description = ProblemDescription(
            target_column='quality_score',
            description='Test quality score problem',
            target_desired_outcome=5
        )
        
        target_field = Field('quality_score', types.int32)
        problem = ClassificationProblem(problem_description, target_field, classes=(1, 2, 3, 4, 5))
        
        # Call questionnaire_prompt_context with correct parameters
        examples = 'Example conversation with quality_score=5\nAnother example with quality_score=2'
        instance_type = 'conversation'
        instance_description = 'customer service conversations'
        n_queries = '3'
        
        context = questionnaire_prompt_context(
            examples=examples,
            problem=problem,
            instance_type=instance_type,
            instance_description=instance_description,
            n_queries=n_queries,
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # Verify the prompt contains expected elements
        assert 'quality_score = 5' in result
        assert 'quality_score != 5' in result
        assert 'up to 3 questions' in result

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_regression_problem_with_direction_up(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with regression problem (direction up)."""
        problem_description = ProblemDescription(
            target_column='revenue',
            description='Test revenue problem',
            target_desired_outcome=RegressionDirection.up
        )
        
        target_field = Field('revenue', types.float64)
        problem = RegressionProblem(problem_description, target_field)
        
        # Call questionnaire_prompt_context with correct parameters
        examples = 'Example 1: revenue=$1000\nExample 2: revenue=$500'
        instance_type = 'sales_call'
        instance_description = 'sales calls with customers'
        n_queries = '4'
        
        context = questionnaire_prompt_context(
            examples=examples,
            problem=problem,
            instance_type=instance_type,
            instance_description=instance_description,
            n_queries=n_queries,
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # Verify the prompt contains expected elements
        assert 'high revenue values' in result
        assert 'low revenue values' in result
        if '{goal_description}' in prompt_template:
            assert 'what leads to higher revenue' in result
        assert 'up to 4 questions' in result
        assert 'sales calls' in result

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_regression_problem_with_direction_down(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with regression problem (direction down)."""
        problem_description = ProblemDescription(
            target_column='response_time',
            description='Test response time problem',
            target_desired_outcome=RegressionDirection.down
        )
        
        target_field = Field('response_time', types.int32)
        problem = RegressionProblem(problem_description, target_field)
        
        # Call questionnaire_prompt_context with correct parameters
        examples = 'Fast response: 30 seconds\nSlow response: 120 seconds'
        instance_type = 'support_ticket'
        instance_description = 'customer support tickets'
        n_queries = '6'
        
        context = questionnaire_prompt_context(
            examples=examples,
            problem=problem,
            instance_type=instance_type,
            instance_description=instance_description,
            n_queries=n_queries,
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # Verify the prompt contains expected elements
        assert 'low response_time values' in result
        assert 'high response_time values' in result
        if '{goal_description}' in prompt_template:
            assert 'what leads to lower response_time' in result
        assert 'up to 6 questions' in result
        assert 'support tickets' in result

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_classification_with_optional_business_domain(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with business domain specified."""
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem with business domain',
            target_desired_outcome='success',
            business_domain='customer service'
        )
        
        # Classes must be sorted in ascending order
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        context = questionnaire_prompt_context(
            examples='Example data',
            problem=problem,
            instance_type='conversation',
            instance_description='conversations',
            n_queries='3',
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # The business_domain should appear in the prompt
        assert 'You are an expert in customer service.' in result
        assert 'target = success' in result

    @pytest.mark.parametrize('prompt_template', [
        QUESTIONNAIRE_PROMPT,
        ACTIONABLE_QUESTIONNAIRE_PROMPT,
        CREATIVE_FEATURES_PROMPT,
    ])
    def test_regression_with_optional_problem_context(self, prompt_template: str) -> None:
        """Test questionnaire prompt creation with problem context specified."""
        problem_description = ProblemDescription(
            target_column='satisfaction',
            target_desired_outcome=RegressionDirection.up,
            name='Customer Satisfaction Analysis',
            description='Analyze factors affecting customer satisfaction scores'
        )
        
        target_field = Field('satisfaction', types.float64)
        problem = RegressionProblem(problem_description, target_field)
        
        context = questionnaire_prompt_context(
            examples='Customer A: satisfaction=4.5\nCustomer B: satisfaction=2.1',
            problem=problem,
            instance_type='customer_interaction',
            instance_description='customer service interactions',
            n_queries='2',
            existing_queries=[]
        )
        result = prompt_template.format(**context)
        
        # The name and description should appear in the prompt
        assert 'Problem: Customer Satisfaction Analysis' in result
        assert 'Description: Analyze factors affecting customer satisfaction scores' in result
        assert 'high satisfaction values' in result
        if '{goal_description}' in prompt_template:
            assert 'what leads to higher satisfaction' in result
        assert 'up to 2 questions' in result


class TestPromptSpecificFeatures:
    """Test specific features unique to each prompt template."""

    def test_actionable_questionnaire_includes_existing_queries(self) -> None:
        """Test that ACTIONABLE_QUESTIONNAIRE_PROMPT includes existing queries section."""
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem',
            target_desired_outcome='success'
        )
        
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        # Create some existing queries
        existing_queries = [
            Query(name='existing_q1', query_text='What is the customer sentiment? (e.g positive, negative, neutral)', return_type=types.string),
            Query(name='existing_q2', query_text='How many products were mentioned? (e.g 1, 2, 3)', return_type=types.int32)
        ]
        
        context = questionnaire_prompt_context(
            examples='Example data',
            problem=problem,
            instance_type='conversation',
            instance_description='conversations',
            n_queries='3',
            existing_queries=existing_queries
        )
        result = ACTIONABLE_QUESTIONNAIRE_PROMPT.format(**context)
        
        # Check that existing queries appear in the prompt
        for query in existing_queries:
            assert query.name in result
            assert query.query_text in result

    def test_actionable_questionnaire_with_no_existing_queries(self) -> None:
        """Test ACTIONABLE_QUESTIONNAIRE_PROMPT with empty existing queries."""
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem',
            target_desired_outcome='success'
        )
        
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        context = questionnaire_prompt_context(
            examples='Example data',
            problem=problem,
            instance_type='conversation',
            instance_description='conversations',
            n_queries='3',
            existing_queries=[]
        )
        result = ACTIONABLE_QUESTIONNAIRE_PROMPT.format(**context)
        
        # Check that it shows "No existing questions" when list is empty
        assert 'No existing questions.' in result
        assert 'actionable questions' in result

    def test_juicy_features_prompt_specific_content(self) -> None:
        """Test that JUICY_FEATURES_PROMPT includes its specific content."""
        problem_description = ProblemDescription(
            target_column='target',
            description='Test problem',
            target_desired_outcome='success'
        )
        
        target_field = Field('target', types.string)
        problem = ClassificationProblem(problem_description, target_field, classes=('failure', 'success'))
        
        # Create some existing queries
        existing_queries = [
            Query(name='existing_q1', query_text='What is the customer sentiment? (e.g positive, negative, neutral)', return_type=types.string),
            Query(name='existing_q2', query_text='How many products were mentioned? (e.g 1, 2, 3)', return_type=types.int32)
        ]
        
        context = questionnaire_prompt_context(
            examples='Example data',
            problem=problem,
            instance_type='conversation',
            instance_description='conversations',
            n_queries='3',
            existing_queries=existing_queries
        )
        result = CREATIVE_FEATURES_PROMPT.format(**context)
        
        # Check that existing queries appear in the prompt
        for query in existing_queries:
            assert query.name in result
            assert query.query_text in result

        # Check for JUICY_FEATURES_PROMPT specific content
        assert 'more "juicy"' in result
        assert 'semantic analysis' in result
        assert 'straight forward' in result
        assert 'not actionable' in result


class TestCreateEnrichConversationPrompt:
    """Test create_enrich_conversation_prompt function."""

    def test_basic_functionality(self) -> None:
        """Test enrich conversation prompt creation with basic parameters."""
        instance = 'Customer: I need help\nAgent: How can I assist you?'
        queries_str = "What is the customer's main issue?"
        instance_description = 'customer service conversations'
        
        result = create_enrich_conversation_prompt(
            instance=instance,
            queries_str=queries_str,
            instance_description=instance_description
        )
        
        assert instance in result
        assert queries_str in result
        assert instance_description in result
        assert f'"{PARSER_OUT_FIELD}"' in result
        assert '```json' in result


class TestGetOutputInstructions:
    """Test get_output_instructions function."""

    def test_boolean_dtype(self) -> None:
        """Test output instructions for boolean dtype."""
        result = get_output_instructions(types.boolean)
        
        assert 'boolean value (true or false)' in result
        assert f'"{PARSER_OUT_FIELD}": true' in result

    def test_int32_dtype(self) -> None:
        """Test output instructions for int32 dtype."""
        result = get_output_instructions(types.int32)
        
        assert 'integer value' in result
        assert f'"{PARSER_OUT_FIELD}": 42' in result

    def test_float64_dtype(self) -> None:
        """Test output instructions for float64 dtype."""
        result = get_output_instructions(types.float64)
        
        assert 'numeric value (integer or float)' in result
        assert f'"{PARSER_OUT_FIELD}": 3.14' in result

    def test_enum_dtype(self) -> None:
        """Test output instructions for enum dtype."""
        enum_dtype = types.EnumDtype('option1', 'option2', 'option3')
        
        result = get_output_instructions(enum_dtype)
        
        assert '"option1", "option2", "option3"' in result
        assert CategoricalFeature.other_category in result
        assert f'"{PARSER_OUT_FIELD}": "option1"' in result

    def test_unsupported_dtype(self) -> None:
        """Test that unsupported dtype raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match='Output instructions not implemented for dtype'):
            get_output_instructions(types.date_dtype)
