from collections.abc import Sequence
from pathlib import Path

from attrs import frozen

from agentune.analyze.api.base import RunContext
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, FloatFeature
from agentune.analyze.join.base import JoinStrategy


@frozen
class _FloatTestFeature(FloatFeature):
    # Redeclare everything with defaults
    name: str = 'FloatFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: float = 10.0
    default_for_nan: float = 11.0
    default_for_infinity: float = 12.0
    default_for_neg_infinity: float = 13.0

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()


# Value to serialize
test_feature = _FloatTestFeature()


async def test_dumps_loads(ctx: RunContext) -> None:
    """Test serialization and deserialization to/from JSON strings."""
    # Test dumps
    json_str = ctx.json.dumps(test_feature)
    assert isinstance(json_str, str)
    assert '_FloatTestFeature' in json_str

    # Test loads with specific type
    loaded_specific = ctx.json.loads(json_str, _FloatTestFeature)
    assert loaded_specific == test_feature
    assert isinstance(loaded_specific, _FloatTestFeature)

    # Test loads with base type - should deserialize to original subclass
    loaded_base = ctx.json.loads(json_str, Feature)  # type: ignore[type-abstract]
    assert loaded_base == test_feature
    assert isinstance(loaded_base, _FloatTestFeature)

    # Test that kwargs are passed through to json.dumps
    json_str_indented = ctx.json.dumps(test_feature, indent=2)
    assert '\n' in json_str_indented
    json_str_compact = ctx.json.dumps(test_feature, separators=(',', ':'))
    assert len(json_str_compact) < len(json_str)


async def test_dump_load_path(ctx: RunContext, tmp_path: Path) -> None:
    """Test serialization and deserialization to/from files using Path."""
    path = tmp_path / 'test_feature.json'

    for file_path in list[str|Path]([path, str(path)]):

        # Test dump to Path
        ctx.json.dump(test_feature, file_path)
        assert Path(file_path).exists()

        # Test load from Path with specific type
        loaded_specific = ctx.json.load(file_path, _FloatTestFeature)
        assert loaded_specific == test_feature
        assert isinstance(loaded_specific, _FloatTestFeature)

        # Test load from Path with base type
        loaded_base = ctx.json.load(file_path, Feature)  # type: ignore[type-abstract]
        assert loaded_base == test_feature
        assert isinstance(loaded_base, _FloatTestFeature)

    # Test that kwargs are passed through
    file_path_indented = tmp_path / 'test_feature_indented.json'
    ctx.json.dump(test_feature, file_path_indented, indent=2)
    content = file_path_indented.read_text()
    assert '\n' in content


async def test_loads_with_kwargs(ctx: RunContext) -> None:
    """Test that kwargs are passed through to json.loads."""
    json_with_number = '{"value": 1.5}'
    result = ctx.json.loads(json_with_number, dict, parse_float=lambda x: float(x) * 2)
    assert result['value'] == 3.0


async def test_dump_load_textio(ctx: RunContext, tmp_path: Path) -> None:
    """Test serialization and deserialization to/from TextIO streams."""
    file_path = tmp_path / 'test_feature.json'

    # Test dump to TextIO (file handle)
    with file_path.open('w') as stream_out:
        ctx.json.dump(test_feature, stream_out)

    json_str = file_path.read_text()
    assert 'FloatTestFeature' in json_str

    # Test load from TextIO with specific type
    with file_path.open() as stream_in:
        loaded_specific = ctx.json.load(stream_in, _FloatTestFeature)
        assert loaded_specific == test_feature
        assert isinstance(loaded_specific, _FloatTestFeature)

    # Test load from TextIO with base type
    with file_path.open() as stream_in:
        loaded_base = ctx.json.load(stream_in, Feature)  # type: ignore[type-abstract]
        assert loaded_base == test_feature
        assert isinstance(loaded_base, _FloatTestFeature)

    # Test that kwargs are passed through for dump
    file_path_indented = tmp_path / 'test_feature_indented.json'
    with file_path_indented.open('w') as stream_out_indented:
        ctx.json.dump(test_feature, stream_out_indented, indent=2)

    content = file_path_indented.read_text()
    assert '\n' in content

    # Test with StringIO
    from io import StringIO
    stringio_out = StringIO()
    ctx.json.dump(test_feature, stringio_out)
    json_str_from_stringio = stringio_out.getvalue()
    assert 'FloatTestFeature' in json_str_from_stringio

    stringio_in = StringIO(json_str_from_stringio)
    loaded_from_stringio = ctx.json.load(stringio_in, _FloatTestFeature)
    assert loaded_from_stringio == test_feature


async def test_binary_stream_rejected(ctx: RunContext, tmp_path: Path) -> None:
    """Test that binary streams are properly rejected."""
    import pytest

    file_path = tmp_path / 'test_feature.json'

    # Try to dump to binary file - should raise TypeError
    with pytest.raises(TypeError, match='Unsupported target type'):
        with file_path.open('wb') as binary_out:
            ctx.json.dump(test_feature, binary_out)  # type: ignore[arg-type]

    # Write valid JSON for load test
    ctx.json.dump(test_feature, file_path)

    # Try to load from binary file - should raise TypeError
    with pytest.raises(TypeError, match='Unsupported source type'):
        with file_path.open('rb') as binary_in:
            ctx.json.load(binary_in, _FloatTestFeature)  # type: ignore[arg-type]
