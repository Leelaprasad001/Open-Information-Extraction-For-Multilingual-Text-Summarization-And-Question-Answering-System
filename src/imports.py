import streamlit as st
import requests
import PyPDF2
import docx2txt
from bs4 import BeautifulSoup
import spacy
from spacy import displacy 
import spacy_streamlit
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch
from googletrans import Translator
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
