# Modified by the Skill.png authors from T1 at commit 22a16b7 (Apache-2.0);
# the changes are listed in skillpng/benchmarks/t1/upstream/VENDORED.md.

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from skillpng.benchmarks.t1.upstream.tools.schema.search_output import SearchResultsOutputModel


class FilterAttractionsInputModel(BaseModel):
    prior_results: SearchResultsOutputModel = Field(
        ..., description="Prior search results by the user."
    )
    city: Optional[str] = Field(
        default=None,
        description="City the user would like to look for an attraction.",
    )
    state: Optional[str] = Field(
        default=None, description="State the user would like to look for an attraction"
    )
    neighborhood: Optional[List[str]] = Field(
        default=None,
        description="List of neighborhoods the user would like to look for an attraction at.",
    )
    type: Optional[List[str]] = Field(
        default=None,
        description="List of attraction types the user would be interested in ",
    )
    attraction_name: Optional[List[str]] = Field(
        default=None,
        description="List of attraction names the user would like to look at",
    )
    model_config = ConfigDict(extra="forbid")
