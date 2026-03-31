from typing import List, Optional
from pydantic import BaseModel, Field

class AdvisorProfile(BaseModel):
    """
    Structured criteria defining an advisor.
    Extracted from the unstructured Word document.
    """
    advisor_id: str = Field(..., description="Unique ID for the advisor")
    name: str = Field(..., description="Name of the advisor")
    expertise: List[str] = Field(
        default_factory=lambda: ["Not Provided"],
        description="List of professional domains or expertise (e.g., Retirement Planning, Tax)"
    )
    target_clients: List[str] = Field(
        default_factory=lambda: ["Not Provided"],
        description="Target demographics (e.g., High Net Worth, Small Business)"
    )
    communication_style: str = Field(
        default="Not Provided",
        description="Communication style descriptor (e.g., Gentle, Direct, Academic)"
    )

class AdvisorDocument(BaseModel):
    """
    Combines the extracted profile, raw text, and vector embedding for an advisor.
    """
    profile: AdvisorProfile
    full_text: str = Field(..., description="Original extracted raw text")
    # Embedding isn't strictly needed in the schema if managed by FAISS, but included for completeness
    
class ParsedUserNeeds(BaseModel):
    """
    Standardized requirements parsed from natural language.
    """
    expertise_needed: List[str] = Field(
        default_factory=list,
        description="List of required domains or expertise. Empty if vague."
    )
    target_clients_needed: List[str] = Field(
        default_factory=list,
        description="Demographics of the user needing matching. Empty if vague."
    )
    communication_preference: Optional[str] = Field(
        default=None,
        description="Preferred communication style. None if vague."
    )
    is_relevant: bool = Field(
        default=True,
        description="True if the user's query is actually about finding a financial advisor/service. False if it is off-topic (e.g. asking for code, recipe, or a dating request)."
    )
    guidance_message: Optional[str] = Field(
        default=None,
        description="If is_relevant is False, generate a polite message telling the user it's off-topic and exactly how they should describe their ideal advisor."
    )

class RecommendationResult(BaseModel):
    """
    The final matched advisor output.
    """
    advisor: AdvisorProfile
    match_score: float = Field(..., description="Combined matching score")
    rationale: str = Field(..., description="AI generated reasoning for the match")
    citations: List[str] = Field(..., description="Excerpts linking the reasoning to raw text or fields")
