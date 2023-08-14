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
env_r = e.loadEnvironment()

e.showEnvionmentProperties(env_r)

env_n = e.loadNormalizedEnv(env_r)

e.loadModel(env_n)

@tapp.get('/action')
async def getAction(query: List[float] = Query(...)):
   return e.getActionPrediction(env_n, query)
