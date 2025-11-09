from typing import Any, override

import polars as pl
from attrs import field, frozen

# Wrappers for Polars series and DFs whose == returns a bool (and not a series or DF of bools)
# allowing those types to be compared for equality when used as dataclass fields.
# They're still not hashable, because it's expensive and we have nowhere to cache the result 
# (instances of these classes are transient, created on-demand by the containing dataclass's __eq__).

@frozen(eq=False, hash=False)
class ComparablePolarsSeries: #noqa: PLW1641 # polars series is not hashable
    series: pl.Series

    @override
    def __eq__(self, other: object) -> bool:
        self.series.hash()
        if isinstance(other, pl.Series):
            return self.series.equals(other)
        elif isinstance(other, ComparablePolarsSeries):
            return self.series.equals(other.series)
        else:
            return False
    
@frozen(eq=False, hash=False)
class ComparablePolarsDataFrame: #noqa: PLW1641 # polars DF is not hashable
    df: pl.DataFrame

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, pl.DataFrame):
            return self.df.equals(other)
        elif isinstance(other, ComparablePolarsDataFrame):
            return self.df.equals(other.df)
        else:
            return False
   
# Use these for conveniently defining attrs fields of series or DF type when the containig class should be comparable.
# If the containing class shoudl not be comparable, pass eq=False (and hash=False) to the define/frozen decorator.
# Creating a dataclass with a series or DF field without any custom eq logic will result in the dataclass's __eq__
# always throwing an error, because Series.__eq__ and DataFrame.__eq__ return a series/DF and not a single bool.

def series_field(**kwargs: Any) -> Any:
    return field(eq=ComparablePolarsSeries, hash=False, **kwargs)

def df_field(**kwargs: Any) -> Any:
    return field(eq=ComparablePolarsDataFrame, hash=False, **kwargs)
