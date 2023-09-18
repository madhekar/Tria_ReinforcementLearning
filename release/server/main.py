from typing import Union, List
from fastapi import FastAPI, Query
from pydantic import BaseModel
import asyncio
from tria_inference import tria_inference_engine as tie

tapp = FastAPI()

# instance of tria interface engine class
e = tie('Tria Inference Engine',
        'TIE serves prediction requests',
        'tria_rl',
        'TriaClimate-v0',
        'tria_a2c_normalized'
         )

# load tria environment
env_r = e.loadEnvironment()
# display current environment variables and properties.
e.showEnvionmentProperties(env_r)
# convert to trined normalized environment
env_n = e.loadNormalizedEnv(env_r)
# load model for the environment 
env_with_stats = e.loadEnvAndModel(env_n)

# request action for set of observations
@tapp.get('/action')
async def getAction(query: List[float] = Query(...)):
   return e.getActionPrediction(env_with_stats, query)
