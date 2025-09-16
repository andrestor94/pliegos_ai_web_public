from typing import List, Optional, Dict
from pydantic import BaseModel

class TimelineItem(BaseModel):
    label: str
    datetime_iso: Optional[str] = None
    raw_text: Optional[str] = None

class Requirement(BaseModel):
    id: Optional[str] = None
    description: str
    requires_upload: bool = False
    delivery_channel: Optional[str] = None

class Guarantee(BaseModel):
    type: str
    required: bool = False
    percentage: Optional[float] = None
    counter_guarantee: Optional[bool] = None
    notes: Optional[str] = None

class BasicInfo(BaseModel):
    process_number: Optional[str] = None
    process_name: Optional[str] = None
    object: Optional[str] = None
    selection_method: Optional[str] = None
    stage: Optional[str] = None
    scope: Optional[str] = None
    currency: Optional[str] = None
    quote_type: Optional[str] = None
    award_type: Optional[str] = None
    document_type: Optional[str] = None
    offers_limit: Optional[str] = None
    reception_place: Optional[str] = None
    offer_validity_days: Optional[int] = None
    legal_framework: Optional[str] = None
    accepts_redetermination: Optional[bool] = None
    requires_payment: Optional[bool] = None
    other_requirements: Optional[str] = None
    accepts_increase_pct: Optional[float] = None
    accepts_extension: Optional[bool] = None
    process_steps: Optional[List[str]] = None

class AmountDuration(BaseModel):
    amount_ars: Optional[float] = None
    receipt_frequency: Optional[str] = None
    estimated_start_text: Optional[str] = None
    duration_text: Optional[str] = None

class MinutesAwards(BaseModel):
    opening_minutes_date: Optional[str] = None
    award_opinion_date: Optional[str] = None
    award_opinion_status: Optional[str] = None

class StructuredPayload(BaseModel):
    basic_info: BasicInfo
    timeline: List[TimelineItem] = []
    min_requirements: Dict[str, List[Requirement]] = {}
    special_clauses: List[Guarantee] = []
    contract_amount_duration: AmountDuration
    minutes_awards: MinutesAwards

class DeepAnalysisSection(BaseModel):
    title: str
    content_html: str
    section_key: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    structured: StructuredPayload
    deep_analysis: List[DeepAnalysisSection]
