# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 07:28:17 2024

@author: user
"""

import os
import haohaninfo
from order_Lo8 import Record
import numpy as np
from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib


html_temp = """
    <div style="background-color:#FFFF00;padding:20px;border-radius:20px">   
    <h1 style="color:white;text-align:center;">金融期末報告</h1>
    <h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading</h2>
    </div>
    """
stc.html(html_temp)







