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
        default="未提供",
        description="Communication style descriptor (e.g., Gentle, Direct, Academic)"
    )
    branch: str = Field(
        default="未提供",
        description="The branch or location the advisor manages (e.g., 新北, 台北)"
    )

class AdvisorDocument(BaseModel):
    """
    Combines the extracted profile, raw text, and vector embedding for an advisor.
    """
    profile: AdvisorProfile
    full_text: str = Field(..., description="Original extracted raw text")
    
    def get_tags_text(self) -> str:
        """Concatenates expertise, target clients, and communication style for vectorization."""
        tags = []
        # Filter out placeholders like "未提供" or "Not Provided"
        expertise = [e for e in self.profile.expertise if e not in ["未提供", "Not Provided"]]
        targets = [t for t in self.profile.target_clients if t not in ["未提供", "Not Provided"]]
        
        tags.extend(expertise)
        tags.extend(targets)
        if self.profile.communication_style not in ["未提供", "Not Provided"]:
            tags.append(self.profile.communication_style)
            
        return ", ".join(tags)
    
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
    branch_needed: Optional[str] = Field(
        default=None,
        description="Preferred branch or location. None if vague."
    )
    investment_experience: Optional[str] = Field(
        default=None,
        description="投資經驗 (1年以下, 1~3年, 3~5年, 5年以上)"
    )
    products_touched: List[str] = Field(
        default_factory=list,
        description="曾接觸商品"
    )
    asset_allocation: List[str] = Field(
        default_factory=list,
        description="目前的資產配置"
    )
    asset_scale: Optional[str] = Field(
        default=None,
        description="預計管理資產規模 (300~1000萬, 1000~3000萬, 3000萬以上)"
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
