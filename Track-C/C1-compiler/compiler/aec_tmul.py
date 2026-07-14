"""TMUL lowering placeholder — C1 scalar subset does not use tensor ops."""
from typing import List
from aec_ir import IRInst

def tmul_lower(instructions: List[IRInst]) -> List[IRInst]:
    return instructions
