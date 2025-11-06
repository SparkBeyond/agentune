import pickle

from attrs import frozen
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from sklearn.externals.array_api_extra.testing import override

from agentune.analyze.llmcache.base import LLMCacheKey
from agentune.analyze.util.asyncmap import KVStore

# ruff: noqa: S301 # allow use of pickle

@frozen
class SerializingKVStore(KVStore[LLMCacheKey, ChatResponse | CompletionResponse]):
    """Hash cache keys and serialize cache values, storing them in a wrapped store which can be out-of-core.

    The values are serialized with pickle, which Pydantic officially supports. This results in larger values
    (at least when most of the size is class data and not large strings) than json, but json doesn't work
    because some ChatResponse fields have type dict[str, Any] with values which are Pydantic models;
    Pydantic serializes these values as json dicts but has no way to recover the original type when deserializing.
    """
    inner: KVStore[bytes, bytes]

    @override
    def __setitem__(self, key: LLMCacheKey, value: ChatResponse | CompletionResponse, /) -> None:
        self.inner[key.long_hash] = pickle.dumps(value)

    @override
    def __delitem__(self, key: LLMCacheKey, /) -> None:
        del self.inner[key.long_hash]

    @override
    def __getitem__(self, key: LLMCacheKey, /) -> ChatResponse | CompletionResponse:
        serialized = self.inner[key.long_hash]
        return pickle.loads(serialized)

    @override
    def __len__(self) -> int:
        return len(self.inner)
