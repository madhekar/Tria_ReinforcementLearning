from typing import Union, List
from fastapi import FastAPI, Query
from pydantic import BaseModel
import asyncio
from tria_inference import tria_inference_engine as tie

tapp = FastAPI()


e = tie('Tria Inference Engine',
                                'TIE serves prediction requests',
                                'tria_rl',
                                'TriaClimate-v0',
                                'tria_a2c_normalized'
                                )
e.loadEnvironment()

e.showEnvionmentProperties()

e.loadNormalizedEnv()

e.loadModel()

@tapp.get('/action')
async def getAction(query: List[float] = Query(...)):
   e.getActionPrediction(query)
   return query
