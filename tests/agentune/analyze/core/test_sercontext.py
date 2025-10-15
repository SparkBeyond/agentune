import logging
from typing import Any

import httpx
import pytest
from attrs import frozen
from cattrs.preconf.json import JsonConverter
from llama_index.llms.openai import OpenAIResponses

from agentune.analyze.core import types
from agentune.analyze.core.database import ArtIndex, DuckdbIndex, DuckdbName, DuckdbTable
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec, SerializationContext
from agentune.analyze.core.types import Dtype, _SerDtype
from agentune.analyze.feature.base import Feature, LlmFeature
from agentune.analyze.feature.gen.insightful_text_generator.features import (
    InsightfulBoolFeature,
    InsightfulCategoricalFeature,
    InsightfulFloatFeature,
    InsightfulIntFeature,
    InsightfulTextFeature,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
)
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.analyze.join.lookup import LookupJoinStrategy
from agentune.analyze.join.timeseries import KtsJoinStrategy
from agentune.analyze.run.base import RunContext

_logger = logging.getLogger(__name__)

@pytest.fixture
def converter(run_context: RunContext) -> JsonConverter:
    return run_context.ser_context.converter

async def test_llm_serialization(httpx_async_client: httpx.AsyncClient) -> None:
    llm_context = LLMContext(httpx_async_client)
    serialization_context = SerializationContext(llm_context)

    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm = llm_context.from_spec(llm_spec)

    @frozen
    class LlmUsingFoobar:
        model: LLMWithSpec

    foobar = LlmUsingFoobar(LLMWithSpec(llm_spec, llm))

    serialized = serialization_context.converter.dumps(foobar)
    recovered = serialization_context.converter.loads(serialized, LlmUsingFoobar)
    assert recovered.model.spec == llm_spec
    assert isinstance(recovered.model.llm, OpenAIResponses)
    assert recovered.model.llm.model == 'gpt-4o'


def test_dtype_serialization(converter: JsonConverter) -> None:
    all_types = [
        *types._simple_dtypes,
        types.EnumDtype('a', 'b', 'c'),
        types.ListDtype(types.int32),
        types.ListDtype(types.EnumDtype('a', 'b', 'c')),
        types.ArrayDtype(types.int32, 3),
        types.ArrayDtype(types.ArrayDtype(types.EnumDtype('a', 'b'), 5), 3),
        types.StructDtype(('a', types.int32), ('b', types.string)),
    ]
    assert converter.unstructure(types.int16) == { '_type': 'simple', 'name': 'int16' }
    assert converter.unstructure(types.EnumDtype('foo', 'bar')) == { '_type': 'enum', 'values': ['foo', 'bar'] }
    for t in all_types:
        assert converter.loads(converter.dumps(t._to_ser_dtype()), _SerDtype) == t._to_ser_dtype() # type: ignore[type-abstract]
        assert converter.loads(converter.dumps(t), Dtype) == t # type: ignore[type-abstract]

def test_index_serialization(converter: JsonConverter) -> None:
    index = ArtIndex(DuckdbName('tab', 'db', 'schema'), ('a', 'b'))
    assert converter.loads(converter.dumps(index), ArtIndex) == index
    assert converter.loads(converter.dumps(index), DuckdbIndex) == index # type: ignore[type-abstract]


def test_join_strategy_serialization(converter: JsonConverter) -> None:
    lookup_strat = LookupJoinStrategy[int]('lookup', DuckdbTable(DuckdbName('tab', 'db', 'schema'), Schema(())), Field('key', types.int32), ())
    kts_strat = KtsJoinStrategy[int]('kts', DuckdbTable(DuckdbName('tab', 'db', 'schema'), Schema(())), Field('key', types.int32), Field('date', types.timestamp), ())
    conv_strat = ConversationJoinStrategy[int]('conv', DuckdbTable(DuckdbName('tab', 'db', 'schema'), Schema(())),
                                               Field('main_id', types.int32), Field('id', types.int32), Field('timestamp', types.timestamp),
                                               Field('role', types.string), Field('content', types.string))

    assert converter.unstructure(lookup_strat) == \
           {'name': 'lookup', 'table': {
               'name': {'name': 'tab', 'database': 'db', 'schema': 'schema'}, 'schema': {'cols': []}, 'indexes': []},
            'key_col': {'name': 'key',
                        'dtype': {'name': 'int32', '_type': 'simple'}},
            'value_cols': [], '_type': 'LookupJoinStrategy'}

    assert converter.loads(converter.dumps(lookup_strat), LookupJoinStrategy[Any]) == lookup_strat
    assert converter.loads(converter.dumps(lookup_strat), LookupJoinStrategy[int]) == lookup_strat
    assert converter.loads(converter.dumps(lookup_strat), LookupJoinStrategy) == lookup_strat
    assert converter.loads(converter.dumps(lookup_strat), JoinStrategy) == lookup_strat # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(kts_strat), KtsJoinStrategy[Any]) == kts_strat
    assert converter.loads(converter.dumps(kts_strat), KtsJoinStrategy[int]) == kts_strat
    assert converter.loads(converter.dumps(kts_strat), KtsJoinStrategy) == kts_strat
    assert converter.loads(converter.dumps(kts_strat), JoinStrategy) == kts_strat # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(conv_strat), ConversationJoinStrategy[Any]) == conv_strat
    assert converter.loads(converter.dumps(conv_strat), ConversationJoinStrategy[int]) == conv_strat
    assert converter.loads(converter.dumps(conv_strat), ConversationJoinStrategy) == conv_strat
    assert converter.loads(converter.dumps(conv_strat), JoinStrategy) == conv_strat # type: ignore[type-abstract]

def test_serialize_feature(converter: JsonConverter, httpx_async_client: httpx.AsyncClient) -> None:
    """Add all real (production) concrete subclasses of Feature here"""
    llm_context = LLMContext(httpx_async_client)
    serialization_context = SerializationContext(llm_context)
    converter = serialization_context.converter

    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm = llm_context.from_spec(llm_spec)
    llm_with_spec = LLMWithSpec(llm_spec, llm)

    conv_strat = ConversationJoinStrategy[int]('conv', DuckdbTable(DuckdbName('tab', 'db', 'schema'), Schema(())),
                                               Field('main_id', types.int32), Field('id', types.int32), Field('timestamp', types.timestamp),
                                               Field('role', types.string), Field('content', types.string))

    data_formatter = ConversationFormatter('name', conv_strat, (Field('a', types.int32),))

    features = [
        InsightfulBoolFeature('name', 'desc', '', llm_with_spec, data_formatter, 'param', 'template', {'a': 'b'}, True),
        InsightfulIntFeature('name', 'desc', '', llm_with_spec, data_formatter, 'param', 'template', {'a': 'b'}, 0),
        InsightfulFloatFeature(name='name', description='desc', technical_description='tech',
                               default_for_missing=0.0, default_for_nan=0.0, default_for_infinity=0.0, default_for_neg_infinity=0.0,
                               model=llm_with_spec, formatter=data_formatter, formatter_param_name='param',
                               prompt_template='template',prompt_parameters={'a': 'b'}),
        InsightfulCategoricalFeature(name='name', description='desc', technical_description='', default_for_missing='a',
                                     categories=('a', 'b'), model=llm_with_spec, formatter=data_formatter, formatter_param_name='param',
                                     prompt_template='template', prompt_parameters= {'a': 'b'}),
    ]

    for feature in features:
        for tpe in [Feature[Any], Feature, InsightfulTextFeature, LlmFeature, type(feature)]:
            assert converter.loads(converter.dumps(feature), tpe) == feature, f'Roundtrip {feature.__class__.__name__} with formal type {tpe}' # type: ignore[arg-type]
