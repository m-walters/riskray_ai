from typing import Optional
from pydicom import valuerep as dicom_type
"""
Functions for converting dicom metadata
"""

def patient_age(x: str) -> Optional[int]:
    # Incoming example: '062Y'
    try:
        return int(x.rstrip('Y'))
    except Exception:
        return None

def patient_sex(x: str) -> float:
    # Incoming may be 'A', 'M', 'F', None
    # TODO -- Double check this
    if x == 'M':
        return 0.
    elif x == 'F':
        return 1.
    else:
        return 0.5

def pregnancy(x: int) -> float:
    # 1: not preg | 2: possibly | 3: definitely | 4: unknown
    # Let's return these on a range of [0,1] (float)
    x = int(x)
    if x == 4:
        return 0.
    else:
        return x / 3.

def dicom_float(x: dicom_type.DSfloat) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def dicom_int(x: int) -> Optional[float]:
    # For consistent datatypes across our meta vector, turn ints into floats
    try:
        return float(x)
    except Exception:
        return None