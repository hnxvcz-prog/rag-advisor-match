# Data Model: Rag Advisor Match MVP

## Entities

### `AdvisorProfile` (Structured Metadata)
- `advisor_id` (str): Unique identifier.
- `name` (str): Name of the advisor.
- `expertise` (List[str]): Extracted professional domain tags (e.g., "Retirement Planning", "Tax").
- `target_clients` (List[str]): Target demographics (e.g., "High Net Worth", "Small Business").
- `communication_style` (str): Style descriptor (e.g., "Gentle", "Direct").
- *Note: Missing fields will default to "Not Provided".*

### `AdvisorDocument` (Unstructured Context)
- `advisor_id` (str): Foreign key to Profile.
- `full_text` (str): Raw text parsed from Word document.
- `embedding` (List[float]): Vector representation of `full_text`.

### `UserQueryContext`
- `raw_input` (str): Natural language string from user.
- `parsed_needs` (Dict):
  - `expertise_needed` (List[str])
  - `target_clients_needed` (List[str])
  - `communication_preference` (str)

### `RecommendationResult`
- `advisor_id` (str)
- `match_score` (float): Combined semantic + structured score.
- `rationale` (str): AI-generated explanation.
- `citations` (List[str]): Snippets or field values verifying the match.
