from pydantic import BaseModel, Field


class CustomerResponse(BaseModel):
    """Customer's response (if relevant) with reasoning."""
    reasoning: str = Field(description="Detailed reasoning for the customer's decision")
    should_respond: bool = Field(description="Whether the customer should respond at this point")
    response: str | None = Field(description="The customer's response if they choose to respond", default=None)