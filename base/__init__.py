from flask import Flask

app = Flask(__name__)

from base.views import homepage
