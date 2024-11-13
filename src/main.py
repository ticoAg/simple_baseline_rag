# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-13 21:32:02
@desc    :   server
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/hello")
async def say_hello():
    return {"message": "Hello World"}
