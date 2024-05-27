import time

def time_function(value):
    m,s,ms = value.split(':')
    return int(m)*60 + int(s) + int(ms)/1000