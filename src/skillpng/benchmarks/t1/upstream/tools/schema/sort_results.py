# Modified by the Skill.png authors from T1 at commit 22a16b7 (Apache-2.0);
# the changes are listed in skillpng/benchmarks/t1/upstream/VENDORED.md.

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from skillpng.benchmarks.t1.upstream.tools.schema.search_output import SearchResultsOutputModel


class SortResultsInputModel(BaseModel):
    results: SearchResultsOutputModel = Field(
        ..., description="Search results that need to be sorted"
    )
    column: str = Field(..., description="Column of the dataset to sort on.")
    ascending: Optional[bool] = Field(
        default=True,
        description="Sorting order either ascending or descending. Defaults to True which is ascending.",
    )
    model_config = ConfigDict(extra="forbid")
