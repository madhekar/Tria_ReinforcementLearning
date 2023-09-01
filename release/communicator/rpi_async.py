import aiohttp
import asyncio

async def main():
    async with aiohttp.ClientSession() as session:
        rlmodel_url = 'http://127.0.0.1:8000/action?query=44&query=89&query=998'
        async with session.get(rlmodel_url) as resp:
            action = await resp.json()
            print('action: ', action['action'])

asyncio.run(main())