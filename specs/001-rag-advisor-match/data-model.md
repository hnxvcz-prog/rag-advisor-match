# Data Model: Rag Advisor Match MVP

## Entities

### `AdvisorProfile` (Structured Metadata)
- `advisor_id` (str): Unique identifier.
- `name` (str): Name of the advisor.
- `branch` (str): The branch or location the advisor manages.
- `expertise` (List[str]): Extracted professional domain tags (e.g., "Retirement Planning", "Tax").
- `target_clients` (List[str]): Target demographics (e.g., "High Net Worth", "Small Business").
- `communication_style` (str): Style descriptor (e.g., "Gentle", "Direct").
- *Note: Missing fields will default to "Not Provided" or "未提供".*

### `AdvisorDocument` (Unstructured Context)
- `advisor_id` (str): Foreign key to Profile.
- `full_text` (str): Raw text parsed from Word document.
- `embedding` (List[float]): Vector representation of `full_text`.

### `UserQueryContext` / `ParsedUserNeeds`
- `raw_input` (str): Natural language string from user (Traits + Experiences).
- `parsed_needs` (Dict):
  - `is_relevant` (bool): Security/Relevance check for off-topic query filtering.
  - `guidance_message` (str): Reply text if `is_relevant` is false.
  - `branch_needed` (str): Preferred branch location.
  - `expertise_needed` (List[str])
  - `target_clients_needed` (List[str])
  - `communication_preference` (str)
  - `investment_experience` (str)
  - `products_touched` (List[str])
  - `asset_allocation` (List[str])
  - `asset_scale` (str)

### `RecommendationResult`
- `advisor_id` (str)
- `match_score` (float): Combined semantic + structured score.
- `rationale` (str): AI-generated explanation.
- `citations` (List[str]): Snippets or field values verifying the match.
