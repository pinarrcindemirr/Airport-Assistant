from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModalityResult:
    modality: str         
    answered: bool           
    confidence: float        
    record_id: str | None    
    raw_score: float        
    extra: dict = field(default_factory=dict)  


@dataclass
class FusionResponse:
    answered: bool
    record: dict | None                
    confidence: float                  
    modality_results: list[ModalityResult]
    agreement: bool | None            
    message: str                     