from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, Literal
from fastapi import FastAPI
from typing import Optional,List
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


class userInput(BaseModel):
    company_reviews: Annotated[float, Field(..., gt=0,description='Enter the company reviews')]
    Reviews: Annotated[float, Field(..., gt=0, description="Enter Reviews in company")]
   #  Salaries: Annotated[float, Field(..., gt=0, description='Enter Avg salary in company')]
    Interviews: Annotated[int, Field(..., gt=0, description='Enter intervices in compay')]
    jobs: Annotated[int, Field(..., gt=0, description='Enter the jobs in company')]
    Benefits: Annotated[int, Field(..., gt=0, description='Enter benefits in company')]
    photos: Annotated[int, Field(..., description='Enter who many photos persent on websites')]
    company_reviews_encoded: Optional[List[float]] = None

    model_config = dict(arbitrary_types_allowed=True)
    @field_validator('company_reviews')
    @classmethod
    def company_reviews(cls, value):
         
         if value > 4:
            return 'excellent'
         elif value < 4 and value > 3:
            return 'good'
         else:
            return 'Poor' 
         
    @model_validator(mode='after')
    @property
    def oneHotEncoder(self):
        encoder = OneHotEncoder(categories=[['excellent','good','Poor']],drop='first',sparse_output=False)
        catogery = np.array(self.company_reviews).reshape(-1,1)
        encoder.fit(np.array(['excellent','good','Poor']).reshape(-1,1))
        self.company_reviews_encoded = encoder.transform(catogery)
        return self
    



